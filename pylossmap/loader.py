import logging
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

from .utils import DB
from .metadata.headers import HEADERS
from .data import BLMData
from .utils import get_ADT
from .utils import to_datetime
from .utils import row_from_time
from .utils import beammode_to_df
from .utils import fill_from_time
from .utils import sanitize_t
from .blm_type_map import name_to_type


class BLMDataFetcher:
    def __init__(self,
                 d_t='30M',
                 BLM_var='LHC.BLMI:LOSS_RS09',
                 pbar=True):
        """This class handles the fetching and loading of loss map data.
        It will query the timber/CALS db for the data and metadata, figure out
        and load the appropriate BLM type & coord mapping.

        Args:
            d_t (str, optional): Due to timber detch size limitations, it
            will be split into chunks of d_t. d_t must be a pd.Timedelta
            format str, e.g. 30M, 1H, 10S, ...
            BLM_var (str, optional): Timber/CALS BLM measurement variable.
            pbar (bool, optional): controls whether to display progessbars.
        """
        self.__header = None
        self.__coord_file = Path(__file__).parent / 'metadata' / 'blm_dcum.csv'

        self.d_t = d_t
        self.BLM_var = BLM_var

        self._logger = logging.getLogger(__name__)
        # header cache
        self._coord_t = self._get_coord_t()
        self._pbar = pbar
        self._db = DB

    def clear_header(self):
        """Clears cached headers.
        """
        self.__header = None

    def from_datetimes(self, t1, t2, keep_headers=False, **kwargs):
        """Create a BLMData instance with data for the requested interval.
        Note: the time interval cannot cross fill boundaries.

        Args:
            t1 (Datetime): interval start.
            t2 (Datetime): interval end.
            keep_headers (optional, bool): Controls whether to clear the
            headers before fetching data.
            **kwargs: pass to BLMData __init__.

        Returns:
            BLMDataCycle: BLMDataCycle instance with the desired data.

        Raises:
            ValueError: if time interval crosses fill boundaries, or no
            data is found.
        """
        if not keep_headers:
            self.clear_header()
        t1 = sanitize_t(t1)
        t2 = sanitize_t(t2)

        meta = self.fetch_meta(t1)
        try:
            fill1 = fill_from_time(t1)
            fill2 = fill_from_time(t2)
        except ValueError:
            fill1 = fill_from_time(t1, fuzzy_t='24H')
            fill2 = fill_from_time(t2, fuzzy_t='24H')

        if fill1 != fill2:
            raise ValueError('Fetch cannot cross fill boundaries... yet...')
        # TODO: investigate this ... why doesn't it find ...
        if fill1 is None:
            raise ValueError(f'Unable to find fill for {t1}.')

        bm_df = beammode_to_df(fill1['beamModes'])
        keep_mask = np.logical_and((bm_df >= t1).any(),
                                   (bm_df <= t2).any())
        keep_bm_df = bm_df.loc[:, keep_mask]
        keep_bm_df = keep_bm_df.applymap(lambda x: np.where(x < t1, t1, x))
        keep_bm_df = keep_bm_df.applymap(lambda x: np.where(x > t2, t2, x))
        if 'context' not in kwargs:
            kwargs['context'] = keep_bm_df

        data = self._fetch_data_bm(keep_bm_df)
        if data is None:
            raise ValueError('No BLM data found.')
        BLM_data = BLMData(data, meta, **kwargs)
        return BLM_data

    def from_fill(self,
                  fill_number,
                  beam_modes='all',
                  unique_beam_modes=False,
                  keep_headers=False,
                  **kwargs):
        """Create a BLMData instance with data for the requested fill and
        beam modes.

        Args:
            fill_number (int): fill of interest.
            beam_modes (str/list, optional): either 'all' to get data for
            all beam modes, or a list of beam modes to only request a subset.
            unique_beam_modes (bool, optional): If a fill contains multiple of
            the same beam mode, they will be uniquified, INJPHYS, INJPHYS -->
            INJPHYS, INJPHYS_2. Setting unique_beam_modes to True and providing
            INJPHYS_2 in beam_modes will select the second beam mode.
            keep_headers (optional, bool): Controls whether to clear the
            headers before fetching data.
            **kwargs: passed to BLMData __init__.

        Returns:
            BLMData: BLMData instance with the desired data.

        Raises:
            ValueError: if no data is found.
        """
        if not keep_headers:
            self.clear_header()

        fill, bm = self._fetch_beam_modes(fill_number=fill_number,
                                          subset=beam_modes,
                                          unique_subset=unique_beam_modes)
        meta = self.fetch_meta(fill['startTime'])
        data = self._fetch_data_bm(bm)
        if data is None:
            raise ValueError('No data found.')
        BLM_data = BLMData(data, meta, context=bm, **kwargs)
        return BLM_data

    def bg_from_ADT_trigger(self,
                            trigger_t,
                            dt_prior='0S',
                            dt_post='2S',
                            look_back='6H',
                            look_forward='5S',
                            max_dt='30S'):
        """Fetches the appropriate background data by looking at the triggers
        of the ADT and figuring out a correct time range, where no triggers
        occured.

        Args:
            trigger_t (Datetime, optional): Time of ADT trigger, if None, will
            take the timestamp of the first value.
            dt_prior (str, optional): time delta prior to adt turn on.
            dt_post (str, optional): time delta post previous adt turn off.
            look_back (str, optional): look back from trigger_t when
            fetching adt trigger data.
            look_forward (str, optional): look forward from trigger_t when
            fetching adt trigger data.
            max_dt (str, optional): max time interval of the background.

        Returns:
            DataFrame: DataFrame containing the background signal.
        """
        if trigger_t is None:
            trigger_t = self.data.index.get_level_values('timestamp')[0]
        else:
            trigger_t = sanitize_t(trigger_t)

        max_dt = pd.Timedelta(max_dt)

        joined = get_ADT(trigger_t - pd.Timedelta(look_back),
                         trigger_t + pd.Timedelta(look_forward))
        joined = joined.fillna(method='ffill')
        if not (joined == 1).any(axis=None):
            raise ValueError('No ADT triggers within time range.')
        # remove consecutive adt on
        joined = joined.fillna(method='ffill')
        joined = joined[~np.logical_and(joined.shift() == joined,
                                        joined == 1).any(axis=1)]

        # find falling edges
        pattern = [1, 0]
        matched_t1 = joined.rolling(len(pattern)).apply(lambda x: all(np.equal(x, pattern)),
                                                        raw=False)
        matched_t1 = matched_t1.sum(axis=1).astype(bool)
        # find rising edges
        pattern = [0, 1]
        matched_t2 = joined.rolling(len(pattern)).apply(lambda x: all(np.equal(x, pattern)),
                                                        raw=False)
        matched_t2 = matched_t2.sum(axis=1).astype(bool)
        t2 = joined[matched_t2].index[-1] + pd.Timedelta(dt_prior)
        try:
            # if there is no adt turn off in the look back data. Fallback on
            # max dt
            t1 = (joined[np.logical_and(matched_t1, joined.index < t2)].index[-1] +
                  pd.Timedelta(dt_post))
        except IndexError:
            t1 = t2 - max_dt
        if t2 - t1 > max_dt:
            t1 = t2 - max_dt

        if t1 > t2:
            raise ValueError(f'Failed to fetch background: {t1} > {t2}')
        else:
            self._logger.info(f'Backgound t1: {t1}')
            self._logger.info(f'Backgound t2: {t2}')
            return self.from_datetimes(t1, t2)

    def iter_from_ADT(self, t1, t2,
                      look_forward='5S',
                      look_back='0S',
                      planes=['H', 'V'],
                      beams=[1, 2],
                      keep_headers=False,
                      yield_background=False,
                      **kwargs):
        """Generator of BLMData instances around ADT blowup triggers.

        Args:
            t1 (Datetime): interval start.
            t2 (Datetime): interval end.
            look_forward (str, optional): Timedelta format string, controls
            how much data after ADT trigger to fetch.
            look_back (str, optional): Timedelta format string, controls how
            much data before ADT trigger to fetch.
            planes (list, optional): ADT trigger planes of interest.
            beams (list, optional): ADT trigger beams of itnerest.
            keep_headers (bool, optional): Controls whether to clear the
            headers before fetching data.
            yield_background (bool, optional): yield both the BLM data nd the
            BLM background.
            **kwargs: passed to BLMData __init__.

        Yields:
            BLMData: BLMData instance with data surrounding the ADT
            trigger.
        """
        if not keep_headers:
            self.clear_header()
        t1 = sanitize_t(t1)
        t2 = sanitize_t(t2)

        joined = get_ADT(t1, t2, planes=planes, beams=beams)
        joined = joined.fillna(method='ffill')
        if not (joined == 1).any(axis=None):
            # TODO: cleanup the pbar thing ...
            raise ValueError('No ADT triggers within time range.')
        for c in joined.columns:
            triggers = joined[(joined[c] == 1)].index.tolist()

            if self._pbar:
                triggers = tqdm(triggers, desc=c)

            for t in triggers:
                # set the BLM_max list to the subset for the chosen beam.
                try:
                    beam_plane_token = c.split('_')[-1]
                    context = {'trigger_t': t,
                               'trigger_beam': beam_plane_token[1],
                               'trigger_plane': beam_plane_token[-1]}
                    BLM_data = self.from_datetimes(t - pd.Timedelta(look_back),
                                                   t + pd.Timedelta(look_forward),
                                                   context=context,
                                                   keep_headers=True)
                    if yield_background:
                        try:
                            BLM_bg = self.bg_from_ADT_trigger(t)
                        except ValueError as e:
                            BLM_bg = None
                            self._logger.warning(e)
                        out = (BLM_data, BLM_bg)
                    else:
                        out = BLM_data
                except ValueError as e:
                    self._logger.warning(f'For {t}: {e}')
                    continue
                yield out

    def _fetch_beam_modes(self, fill_number, **kwargs):
        '''Gets the beam mode timing data.

        Args:
            fill_number (int): fill of interest.
            **kwargs: forwarded to utils.beammode_to_df

        Returns:
            tuple (dict, DataFrame): dict containing Datetime of fill start
            time & fill end ts, DataFrame of beam mode start & end.
        '''
        bm_t = self._db.getLHCFillData(fill_number=fill_number)
        # put fill start and end into dict
        fill_t = {}
        fill_t['startTime'] = to_datetime(bm_t['startTime'])
        fill_t['endTime'] = to_datetime(bm_t['endTime'])
        # put beam mode ts into dataframe
        bm_df = beammode_to_df(bm_t['beamModes'], **kwargs)
        return fill_t, bm_df

    def _get_coord_t(self):
        """Extract the timestamps from the coord files in "blm_coords" folder,
        and create a dataframe which maps the timestamp to the file.

        Returns:
            DataFrame: DataFrame with as index the timestamp and one column,
            the file path.
        """
        header_coords = pd.read_csv(self.__coord_file,
                                    index_col=[0, 1],
                                    parse_dates=[0],
                                    date_parser=lambda x: pd.to_datetime(x).tz_convert('Europe/Zurich'))
        # header_coords = pd.read_hdf(self.__coord_file)
        return header_coords

    @staticmethod
    def _date_chunk(start, end, diff):
        '''Create time chunks of size diff, between start and end.
        Note: the start and end times will always be included.

        Args:
            start (Datetime): start time.
            end (Datetime): end time.
            diff (Timedelta): desired time chunks.

        Yields:
            Datetime: chunk boundary.
        '''
        intv = (end - start) // diff
        yield start
        for i in range(1, intv):
            yield (start + diff * i)
        yield end

    def fetch_meta(self, t):
        """Gets the metadata corresponding to the requested timestamp.

        Args:
            t (Datetime): Time of data.

        Returns:
            DataFrame: DataFrame containing blm, position and type.
        """
        meta_time_ind = self._coord_t.index.levels[0].get_loc(t,
                                                              method='ffill')
        meta_time = self._coord_t.index.levels[0][meta_time_ind]
        self._logger.info(f'Using meta from {meta_time}.')
        # this copy is to avoid pandas chain assignment warning
        meta = self._coord_t.loc[meta_time].copy()
        meta['type'] = meta.apply(lambda x: name_to_type(x['blm']),
                                  axis=1)
        meta = meta.set_index('blm')
        return meta

    def fetch_logging_header(self, t):
        """Fetch the fill's column names from independent logging headers.

        Args:
            t (DateTime): Time of data.

        Returns:
            list: list of columns containing the name of BLMs.
        """
        blms = self.fetch_meta(t=t).index.tolist()
        blms = [b for b in blms if b.split('.')[0] not in ['BLMTS', 'BLMMI',
                                                           'BLMES', 'BLMDS',
                                                           'BLMCK', 'BLMAS',
                                                           'BLMCD']]
        return blms

    def fetch_force_header(self, t):
        """Fetch the fill's column names from header.py file.

        Args:
            t (DateTime): Time of data.

        Returns:
            list: list of columns containing the name of BLMs.
        """
        # TODO: this order of these headers can be slightly off...
        headers = pd.DataFrame.from_dict(HEADERS, orient='index')
        headers.index = pd.to_datetime(headers.index).tz_localize('Europe/Zurich')
        headers = headers.sort_index()
        headers.index.name = 'timestamp'
        return row_from_time(headers, t, method='ffill').dropna().tolist()

    def fetch_timber_header(self, t):
        """Fetch the fill's column names from timber.

        Args:
            t (Datetime): time of data.

        Returns:
            list: list of columns containing the name of BLMs.
        """
        metadata = self._db.getMetaData(self.BLM_var)[self.BLM_var]
        metadata = pd.DataFrame(list(metadata[1]),
                                index=to_datetime(metadata[0]))
        metadata.index.name = 'timestamp'
        columns = row_from_time(metadata, t,
                                method='ffill').dropna().tolist()
        return columns

    def _fetch_data_t(self, t1, t2, header_fetchers=None):
        """Fetch the data for one time chunk.

        Args:
            t1 (Datetime): Start time of the data fetch.
            t2 (Datetime): End time of the data fetch.
            header_fetchers (list, optional): list of callable which return a
            list. Each is tried in sequence, until a compatible header is
            found.

        Returns:
            DataFrame: DataFrame containing the BLM data.
        """
        if header_fetchers is None:
            header_fetchers = [self.fetch_timber_header,
                               self.fetch_force_header,
                               self.fetch_logging_header]

        data = self._db.get(self.BLM_var, t1, t2)[self.BLM_var]
        if data[1].size == 0:
            return

        if self.__header is None:
            for fetcher in header_fetchers:
                columns = fetcher(t1)
                self._logger.info(f'Header from self.{fetcher.__name__}: {len(columns)}.')
                # self._logger.debug(f'Header from self.{fetcher.__name__}:\n{columns}')
                self._logger.info(f'Number of columns in data: {data[1].shape[1]}.')
                if len(columns) == data[1].shape[1]:
                    self._logger.info(f'Using header from self.{fetcher.__name__}.')
                    self.__header = columns

        if self.__header is None:
            self._logger.error("No compatible header found.")

        data = pd.DataFrame(data[1],
                            index=to_datetime(data[0]),
                            columns=self.__header)
        data.index.name = 'timestamp'
        return data

    def _fetch_data_bm(self, beam_modes):
        """Fetch data for beam modes.

        Args:
            beam_modes (DataFrame): DataFrame containing beam modes as
            columns and "startTime", "endTime" as index.

        Returns:
            DataFrame: MultiIndex DataFrame containing the BLM data for the
            beam modes.
        """
        # TODO: cleanup the pbar thing ...
        data = []

        bm_iter = beam_modes.columns
        if self._pbar and len(beam_modes.columns) > 1:
            bm_iter = tqdm(beam_modes.columns, desc='Beam mode')

        for mode in bm_iter:
            if self._pbar and len(beam_modes.columns) > 1:
                bm_iter.set_description(f'Beam mode {mode}')
            t_chunks = list(self._date_chunk(beam_modes[mode]['startTime'],
                                             beam_modes[mode]['endTime'],
                                             pd.Timedelta(self.d_t)))
            d = []

            t_iter = zip(t_chunks, t_chunks[1:])
            if self._pbar and len(t_chunks) - 1 > 1:
                t_iter = tqdm(t_iter, total=len(t_chunks) - 1)

            for t1, t2 in t_iter:
                if self._pbar and len(t_chunks) - 1 > 1:
                    t_iter.set_description(f'{t1.strftime("%m-%d %H:%M:%S")} ▶{t2.strftime("%m-%d %H:%M:%S")}')
                d_i = self._fetch_data_t(t1, t2)
                if d_i is not None:
                    d.append(d_i)
            if d:
                d = pd.concat(d)
                data.append(d)

        if not data:
            return

        data = pd.concat(data,
                         keys=list(beam_modes.columns),
                         names=['mode']).sort_index(level='timestamp')
        return data
