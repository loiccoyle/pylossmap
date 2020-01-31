import logging
import numpy as np
import pandas as pd

from pathlib import Path
from functools import partial

from .utils import DB
from .utils import BEAM_META
from .utils import row_from_time
from .utils import sanitize_t
from .lossmap import LossMap
from .plotting import plot_waterfall
from .timber_vars import PRIMARY_BLM_7

# TODO: it would be quite cool to have the lossmap filtering and processing
# methods also work for the BLMData object. Maybe the LossMap object should be
# merged with the BLMData object ?


class BLMData:

    @classmethod
    def load(cls, file_path, **kwargs):
        """Load the data from a hdf file and create a BLMData instance.

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
        header = pd.read_hdf(file_path, 'header')
        header = header[0].tolist()
        # # read real columns from csv file & replace fake columns
        # with open(file_path.with_suffix('.csv'), 'r') as fp:
        #     columns = fp.readlines()
        data.columns = [c.rstrip() for c in header]
        return cls(data, meta, **kwargs)

    def __init__(self,
                 data,
                 meta,
                 BLM_filter=r'BLM[Q|B|A|T|2|E]I.*',
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

        self.df = data
        if BLM_filter is not None:
            if not isinstance(BLM_filter, list):
                BLM_filter = [BLM_filter]
            for f in BLM_filter:
                self.df = self.df.filter(regex=f, axis='columns')
        self.meta = self._get_metadata(meta)

        # Dynamically add beam meta fetching methods
        for k, (v, series) in BEAM_META.items():
            meth = partial(self.fetch_var, v, timeseries=series)
            name = f'fetch_{k}'
            meth.__name__ = name
            meth.__doc__ = (f"Gets {k} data from timber for the current "
                            "time range.\n"
                            "Args:\n"
                            # "\tv (str, optional): Timber variable.\n"
                            # "\ttimeseries (bool, optional): whether Timber "
                            # "variable returns a timeseries.\n"
                            "\treturn_raw (bool, optional): if True, returns "
                            "the timestamps along with the data.\n"
                            "\n"
                            "\tkwargs: any timber variable flags, such as the "
                            " beam or the plane."
                            "Returns:\n"
                            "\tDataFrame or tuple: Dataframe with timestamp"
                            "and data if return_raw is True, a tuple "
                            "containing timestamp and data arrays.")
            setattr(self, name, meth)

    def fetch_var(self, v, timeseries=True, return_raw=False, **kwargs):
        """Gets provided timber variable data from timber for the current time range.

        Args:
            v (str, optional): Timber variable.
            timeseries (bool, optional): whether Timber variable returns a
                timeseries.
            return_raw (bool, optional): if True, returns the timestamps along
                with the data.
        Returns:
            DataFrame or tuple: Dataframe with timestamp and data. If
                return_raw is True, a tuple containing timestamp and data
                arrays.
        """
        t1 = self.df.index.get_level_values('timestamp')[0]
        t2 = None
        # key, timeseries = BEAM_META[v]
        if timeseries:
            t2 = self.df.index.get_level_values('timestamp')[-1]
        try:
            v = v.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Provide {e} kwarg.")
        out = DB.get(v, t1, t2)[v]
        if not return_raw:
            out = pd.DataFrame(np.vstack(out).T, columns=['timestamp', v])
            out['timestamp'] = pd.to_datetime(out['timestamp'],
                                              unit='s',
                                              utc=True).dt\
                .tz_convert('Europe/Zurich')
            out = out.set_index('timestamp')
        return out

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

        maxes = self.df.groupby('mode')[BLM_max].idxmax()
        # tuples are (beam_mode, timestamp)
        # only keep timestamp
        return maxes.applymap(lambda x: x[1])

    def iter_max(self, BLM_max=None):
        """Creates an generator of ((mode, datetime), BLM_max), where mode and
        datetime correspond to the max value of BLM_max.

        Args:
            BLM_max (list, optional): List of BLMs, defaults to the primary
                blms in IR 7.

        Returns:
            generator: ((mode, datetime), BLM_max)

        Examples:
            BLM_data = BLMData(data, meta)
            for idx, blms in BLM_date.iter_max():
               row = BLM_data.loc[idx]
        """
        iterable = []
        for mode, row in self.find_max(BLM_max).iterrows():
            iterable.extend([[mode, t, blm] for blm, t in row.items()])
        maxes = pd.DataFrame(iterable, columns=['mode', 'timestamp', 'blm'])\
            .groupby(['mode', 'timestamp'], sort=False)
        return ((r[0], r[1]['blm'].tolist()) for r in maxes)

    def _get_metadata(self, meta):
        """Gets the coords and type for the blms in the "df" DataFrame, for blms
        not found in meta, sets the type to "other" and coord to None.

        Args:
            meta (DataFrame): DataFrame containing the metadata.

        Returns:
            DataFrame: DataFrame with blms as index and "dcum" & "type" as
                columns.
        """
        blms = set(self.df.columns)
        with_meta = list(blms & set(meta.index.tolist()))
        without_meta = list(blms - set(meta.index.tolist()))
        with_meta_df = meta.loc[with_meta]
        without_meta_df = pd.DataFrame({'blm': without_meta}).set_index('blm')
        without_meta_df['type'] = 'other'
        without_meta_df['dcum'] = None
        return pd.concat([with_meta_df, without_meta_df], sort=False)\
            .sort_values('dcum')\
            .astype({'dcum': 'int64', 'type': 'str'})

    def loss_map(self,
                 datetime=None,
                 row=None,
                 context=None,
                 background=None,
                 **kwargs):
        """Creates a LossMap instance.

        Args:
            datetime (Datetime, optional): If provided, is used to find the
                row in the data closest to datetime.
            row (Series, optional): Row of data for which to create the LossMap
                instance.
            context (optional): if None, will use self.context.
            background (Series, optional): if provided will create a LossMap
                instance for the background and set is as the background of the
                returned LossMap instance.
            **kwargs: passed to LossMap.__init__.

        Returns:
            LossMap: LossMap instance of the desired data.

        Raises:
            ValueError: If neither t nor row is provided.
        """
        if datetime is None and row is None:
            raise ValueError('Provide either "datetime", or "row".')
        if row is None:
            row = row_from_time(self.df, datetime,
                                flatten=True, method='nearest')
        if datetime is None:
            try:
                # try to get a datetime from the name of the row being passed.
                # usually (beam_mode, datetime)
                if isinstance(row.name[1], pd.Timestamp):
                    datetime = row.name[1]

                elif isinstance(row.name[0], pd.Timestamp):
                    datetime = row.name[0]
            except (IndexError, TypeError):
                pass
        else:
            datetime = sanitize_t(datetime)

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

        # remove BLM names from columns due to hdf header size limitation.
        # save_data = self.df.copy()
        header = self.df.columns
        try:
            # replace real header with numbers to not have problem saving large
            # header to hdf
            self.df.columns = range(self.df.shape[1])
            self.df.to_hdf(file_path, key='data', format='table')
            self.meta.to_hdf(file_path, key='meta', format='table',
                             append=True)
            pd.DataFrame(header).to_hdf(file_path, key='header',
                                        format='table', append=True)
        finally:
            # put real header back
            self.df.columns = header
        # # write columns real columns name in separate file.
        # with open(file_path.with_suffix('.csv'), 'w') as fp:
        #     fp.write('\n'.join(self.df.columns))

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
            data = self.df
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

    def __getitem__(self, key):
        '''
        Args:
            key (int, float, str): if int then the corresponding data row is
                used to create a LossMap instance. If float or str then assumed
                unix time or pd.to_datetime compatible str.

        Returns:
            LossMap: LossMap instance with the desired BLM data.
        '''
        if isinstance(key, int):
            # get row time from data index.
            time = self.df.index[key][-1]
        else:
            # assume epoch or to_datetime str.
            time = sanitize_t(key)
        return self.loss_map(datetime=time)

    def __repr__(self):
        out = 'df:\n' + self.df.__repr__() + '\nmeta:\n' + self.meta.__repr__()
        if self.context is not None:
            out += '\ncontext:\n' + self.context.__repr__()
        return out
