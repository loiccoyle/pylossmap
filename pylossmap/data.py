import logging
import numpy as np
import pandas as pd

from pathlib import Path

# avoid circular import problem
from .utils import DB
from .utils import BLM_MAX
from .utils import get_ADT
from . import loader
from .lossmap import LossMap
from .lossmap import CollLossMap
from .utils import row_from_time
from .utils import fill_from_time
from .utils import beammode_to_df
from .plotting import plot_waterfall


class BLMData:
    def __init__(self,
                 data,
                 meta,
                 BLM_filter=r'BLM[Q|B|A|T|2|E]I*',
                 BLM_max=None,
                 info=None):
        """This class handles the parsing/preprocessing & plotting of the BLM data.

        Args:
            data (DataFrame): MultiIndex DataFrame containing the BLM
            measurements, for the various beam modes & query chunks.
            meta (DataFrame): DataFrame containing the BLM metadata.
            BLM_filter (list/str, optional): regex str of list of regex strs,
            BLMs of interest.
            BLM_max (list, optional): list of BLM names for which to
            find the max values. Defaults to TCP collimators.
            info (optional): additional info.
        """
        self._logger = logging.getLogger(__name__)
        self.info = info
        if BLM_max is None:
            BLM_max = BLM_MAX
        self.BLM_max = BLM_max

        self.data = data
        if BLM_filter is not None:
            if not isinstance(BLM_filter, list):
                BLM_filter = [BLM_filter]
            for f in BLM_filter:
                self.data = self.data.filter(regex=f, axis='columns')
        self.meta = self.get_metadata(meta)

    def find_max(self):
        """Finds the max timestamp and chunk in which the max occured.

        Returns:
            DataFrame: DataFrame containing a tuple: (mode, timestamp).
        """

        maxes = self.data.groupby('mode')[self.BLM_max].idxmax()
        # tuples are (beam_mode, timestamp)
        # only keep timestamp
        return maxes.applymap(lambda x: x[1])

    def iter_maxes(self):
        """Creates an generator of ((mode, datetime), BLM_max), where mode and
        datetime correspond to the max value of BLM_max.

        Returns:
            generator: ((mode, datetime), BLM_max)
        """
        iterable = []
        for mode, row in self.find_max().iterrows():
            iterable.extend([[mode, t, blm] for blm, t in row.items()])
        maxes = pd.DataFrame(iterable, columns=['mode', 'timestamp', 'blm'])\
            .groupby(['mode', 'timestamp'], sort=False)
        return ((r[0], r[1]['blm'].tolist()) for r in maxes)

    def get_metadata(self, meta):
        """Gets the coords and type for the blms in the "data" DataFrame, for blms
        not found in meta, sets the type to "other" and coord to None.

        Args:
            meta (DataFrame): DataFrame containing the metadata.

        Returns:
            DataFrame: DataFrame with blms as index and "dcum" & "type" as
            columns.
        """
        blms = set(self.data.columns)
        with_meta = list(blms & set(meta['blm']))
        without_meta = list(blms - set(meta['blm']))
        with_meta_df = meta.set_index('blm').loc[with_meta]
        without_meta_df = pd.DataFrame({'blm': without_meta}).set_index('blm')
        without_meta_df['type'] = 'other'
        without_meta_df['dcum'] = None 
        return pd.concat([with_meta_df, without_meta_df], sort=False).sort_values('dcum')

    def get_bg_ADT(self, t=None, dt_prior='0S', dt_post='3S', look_back='2H'):
        """Fetches the appropriate background data by looking at the triggers
        of the ADT and figuring out a correct time range, where no triggers
        occured.

        Args:
            t (Datetime, optional): Time of data, if None, will take the timestamp of
            the first value.
            dt_prior (str, optional): time delta prior to adt turn on.
            dt_post (str, optional): time delta post previous adt turn off.
            look_back (str, optional): look back from t when fetching adt
            trigger data.

        Returns:
            DataFrame: DataFrame containing the background signal.
        """
        if t is None:
            t = self.data.index.get_level_values('timestamp')[0]
        joined = get_ADT(t - pd.Timedelta(look_back), t)
        joined = joined.fillna(method='ffill')
        if not (joined == 1).any(axis=None):
            self._logger.warning('No ADT triggers within time range.')
            return None

        self._logger.debug(joined)
        # get the latest times after previous adt off and before adt on.
        t2 = (joined[(joined == 1).any(axis=1)].index[-1] -
              pd.Timedelta(dt_prior))
        joined = joined.loc[joined.index < t2]
        prev_trigger = (joined == 1).any(axis=1).idxmax()
        joined = joined.loc[prev_trigger:]
        self._logger.debug(joined)
        t1 = (joined[(joined == 0).all(axis=1)].index[0] +
              pd.Timedelta(dt_post))

        if t1 > t2:
            self._logger.warning('Failed to fetch background.')
            bg = None
        else:
            self._logger.info(f'Backgound t1: {t1}')
            self._logger.info(f'Backgound t2: {t2}')
            data = self.data.droplevel(0)
            mask = np.logical_and(data.index >= t1, data.index <= t2)
            bg = data[mask]
        return bg

    def get_bg_INJPROT(self):

        if 'INJPROT' in self.data.index.get_level_values('mode'):
            inj_prot_df = self.data.loc['INJPROT']
        else:
            self._logger.info('INJPROT not in current data, will fetch from Timber.')
            t = self.data.index.get_level_values('timestamp')[0]
            fill = fill_from_time(t)
            inj_prot_df = loader.LossMapFetcher().from_fill(fill['fillNumber'],
                                                            beam_modes=['INJPROT']).data
            inj_prot_df = inj_prot_df.loc['INJPROT']
        return inj_prot_df

    def loss_map(self,
                 t=None,
                 row=None,
                 background=None,
                 **kwargs):
        """Creates a LossMap instance.

        Args:
            t (Datetime, optional): If provided, is used to find a desired row in
            the data which corresponds to t.
            row (Series, optional): Row of data for which to create the LossMap
            instance.
            background (DataFrame, optional): BLM data background data.
            **kwargs: Description

        Returns:
            LossMap: LossMap instance of the desired data.

        Raises:
            ValueError: If neither t nor row is provided.
        """
        # TODO: figure out what to do with background, do I want to fetch it ?
        if t is None and row is None:
            raise ValueError('Provide either "t", or "row".')
        if row is None:
            row = row_from_time(self.data, t, flatten=True, method='nearest')
            row.name = 'data'
        # if isinstance(background, str) and background == 'fetch':
        #     background = self.get_bg_ADT(t)

        data = pd.concat([row, self.meta], axis=1, sort=False)
        # TODO: make this line obsolete
        data.columns = ['data', 'dcum', 'type']
        return LossMap(data=data,
                       background=background,
                       datetime=t,
                       **kwargs)

    # def coll_loss_map(self, **kwargs):
    #     # TODO: get more coll db files to cover more timestamps.
    #     lossmap = self.loss_map(**kwargs)
    #     return CollLossMap(lossmap) 

    def save(self, file_path):
        """Save the DataFrame to hdf file/keys,

        Args:
            file_path (str/path): Path to hdf file in which to save the
            DataFrames.

        Raises:
            OSError: If file already exists.
        """
        file_path = Path(file_path).with_suffix('.h5')
        if file_path.is_file():
            raise OSError(f'File {file_path} already exists.')

        self._data_bk.to_hdf(file_path, key='data')
        self._meta_bk.to_hdf(file_path, key='meta')

    def plot(self, data=None, **kwargs):
        if data is None:
            data = self.data
        plot_waterfall(data=self.data, meta=self.meta, **kwargs)


def load(file_path):
    """Load the data from a hdf file and create a LossMapData instance.

    Args:
        file_path (str/path): Path to hdf file from which to load the data.

    Returns:
        LossMapData: LossMapData instance with the loaded data.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    file_path = Path(file_path).with_suffix('.h5')
    if not file_path.is_file():
        raise FileNotFoundError(f'File {file_path} not found.')

    data = pd.read_hdf(file_path, 'data')
    meta = pd.read_hdf(file_path, 'meta')

    return BLMData(data, meta, meta)
