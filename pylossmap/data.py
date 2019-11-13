import logging
import numpy as np
import pandas as pd

from pathlib import Path

from .utils import DB
from .utils import BEAM_META
from .utils import row_from_time
from .lossmap import LossMap
from .plotting import plot_waterfall
from .timber_vars import PRIMARY_BLM_7


class BLMData:
    def __init__(self,
                 data,
                 meta,
                 BLM_filter=r'BLM[Q|B|A|T|2|E]I*',
                 context=None):
        """This class handles the parsing/preprocessing & plotting of the BLM data.

        Args:
            data (DataFrame): MultiIndex DataFrame containing the BLM
            measurements, for the various beam modes & query chunks.
            meta (DataFrame): DataFrame containing the BLM metadata.
            BLM_filter (list/str, optional): regex str of list of regex strs,
            BLMs of interest.
            context (optional): additional info.
        """
        self._logger = logging.getLogger(__name__)
        self.context = context

        self.data = data
        if BLM_filter is not None:
            if not isinstance(BLM_filter, list):
                BLM_filter = [BLM_filter]
            for f in BLM_filter:
                self.data = self.data.filter(regex=f, axis='columns')
        self.meta = self._get_metadata(meta)

    def find_max(self, BLM_max=None):
        """Finds the max timestamp and chunk in which the max occured.

        Args:
            BLM_max (list, optional): List of BLMs, defaults to the primary
            blms in IR 7.

        Returns:
            DataFrame: DataFrame containing a tuple: (mode, datetime).
        """
        if BLM_max is None:
            BLM_max = PRIMARY_BLM_7[1] + PRIMARY_BLM_7[2]

        maxes = self.data.groupby('mode')[BLM_max].idxmax()
        # tuples are (beam_mode, timestamp)
        # only keep timestamp
        return maxes.applymap(lambda x: x[1])

    def iter_max(self, **kwargs):
        """Creates an generator of ((mode, datetime), BLM_max), where mode and
        datetime correspond to the max value of BLM_max.

        Example:
            BLM_data = BLMData(data, meta)
            for idx, blms in BLM_date.iter_max():
               row = BLM_data.loc[idx]

        Returns:
            generator: ((mode, datetime), BLM_max)
        """
        iterable = []
        for mode, row in self.find_max(**kwargs).iterrows():
            iterable.extend([[mode, t, blm] for blm, t in row.items()])
        maxes = pd.DataFrame(iterable, columns=['mode', 'timestamp', 'blm'])\
            .groupby(['mode', 'timestamp'], sort=False)
        return ((r[0], r[1]['blm'].tolist()) for r in maxes)

    def _get_metadata(self, meta):
        """Gets the coords and type for the blms in the "data" DataFrame, for blms
        not found in meta, sets the type to "other" and coord to None.

        Args:
            meta (DataFrame): DataFrame containing the metadata.

        Returns:
            DataFrame: DataFrame with blms as index and "dcum" & "type" as
            columns.
        """
        blms = set(self.data.columns)
        with_meta = list(blms & set(meta.index.tolist()))
        without_meta = list(blms - set(meta.index.tolist()))
        with_meta_df = meta.loc[with_meta]
        without_meta_df = pd.DataFrame({'blm': without_meta}).set_index('blm')
        without_meta_df['type'] = 'other'
        without_meta_df['dcum'] = None
        return pd.concat([with_meta_df, without_meta_df],
                         sort=False).sort_values('dcum')

    def get_beam_meta(self, key, **kwargs):
        """Fetched beam meta data from timber.

        Args:
            key (str): a key of utils.BEAM_META.
            **kwargs: beam/plane if the requested timber variable requires it.

        Returns:
            DataFrame: Fetched timber data.

        Raises:
            KeyError: if timber variable requires additional kwargs.
            ValueError: if key is not in utils.BEAM_META.
        """
        if key not in BEAM_META.keys():
            raise ValueError(f'key: "{key}" is not in {BEAM_META.keys()}.')

        t1 = self.data.index.get_level_values('timestamp')[0]
        t2 = None
        key, timeseries = BEAM_META[key]
        if timeseries:
            t2 = self.data.index.get_level_values('timestamp')[-1]
        try:
            key = key.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Provide {e} kwarg.")
        data = DB.get(key, t1, t2)[key]
        out = pd.DataFrame(np.vstack(data).T, columns=['timestamp', key])
        out['timestamp'] = pd.to_datetime(out['timestamp'],
                                          unit='s',
                                          utc=True).dt\
            .tz_convert('Europe/Zurich')
        return out.set_index('timestamp')

    def loss_map(self,
                 datetime=None,
                 row=None,
                 context=None,
                 background=None,
                 **kwargs):
        """Creates a LossMap instance.

        Args:
            datetime (Datetime, optional): If provided, is used to find a
            desired row in the data which corresponds to datetime.
            row (Series, optional): Row of data for which to create the LossMap
            instance.
            context (optional): if None, will use self.context.
            background (Series, optional): if provided will create a LossMap
            instance for the background and will link it to the data's LossMap.
            **kwargs: passed to LossMap.__init__.

        Returns:
            LossMap: LossMap instance of the desired data.

        Raises:
            ValueError: If neither t nor row is provided.
        """
        if datetime is None and row is None:
            raise ValueError('Provide either "datetime", or "row".')
        if row is None:
            row = row_from_time(self.data, datetime,
                                flatten=True, method='nearest')
        if datetime is None:
            try:
                # try to get a datetime from the name of the row being passed.
                # usually (beam_mode, datetime)
                if isinstance(row.name[1], pd.Datetime):
                    datetime = row.name[1]
            except (IndexError, TypeError):
                pass

        if context is None:
            context = self.context

        data = pd.concat([row, self.meta], axis=1, sort=False)
        data.columns = ['data', 'dcum', 'type']
        LM = LossMap(data=data, datetime=datetime, context=context, **kwargs)

        if background is not None:
            background = pd.concat([background, self.meta], axis=1, sort=False)
            background.columns = ['data', 'dcum', 'type']
            background = LossMap(data=background, context=context)
            LM.set_background(background)
        return LM

    def save(self, file_path):
        """Save the DataFrames to hdf file/keys.

        Args:
            file_path (str/path): Path to hdf file in which to save the
            DataFrames.

        Raises:
            OSError: If file already exists.
        """
        file_path = Path(file_path).with_suffix('.h5')
        if file_path.is_file():
            raise OSError(f'File {file_path} already exists.')

        self.data.to_hdf(file_path, key='data')
        self.meta.to_hdf(file_path, key='meta')

    def plot(self, data=None, title=None, **kwargs):
        """Plots a waterfall plot of the data. Note, will produce multiple
        figures if data contains a mode index.

        Args:
            data (DataFrame, optional): DataFrame containing the BLM data.
            title (str, optional): figure title, '{mode}' gets replaced with
            the beam mode.
            **kwargs: passed to plotting.plot_waterfall
        """
        # TODO: fix this format mode thing
        if data is None:
            data = self.data
        if 'mode' in data.index.names:
            out = {}
            for mode, d in data.groupby('mode'):
                if title is not None:
                    kwargs['title'] = title.format(mode=mode)
                else:
                    kwargs['title'] = mode

                out[mode] = plot_waterfall(data=d, meta=self.meta, **kwargs)
            return out
        else:
            return plot_waterfall(data=data, meta=self.meta, **kwargs)


def load(file_path, **kwargs):
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

    return BLMData(data, meta, meta, **kwargs)
