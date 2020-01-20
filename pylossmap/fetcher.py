import logging
import itertools
import numpy as np
import pandas as pd

from pathlib import Path

from .utils import DB
from .timber_vars import INTENSITY
from .data import BLMData
from .utils import get_ADT
from .utils import to_datetime
from .utils import row_from_time
from .utils import beammode_to_df
from .utils import fill_from_time
from .utils import sanitize_t
from .utils import PBar
from .utils import files_to_df
from .blm_type_map import name_to_type


# TODO: fix the progress bar, add return raw option.


class BLMDataFetcher:
    def __init__(self,
                 d_t='30M',
                 BLM_var='LHC.BLMI:LOSS_RS09',
                 pbar=True,
                 mute=False):
        """This class handles the fetching and loading of BLM data.
        It will query the timber/CALS db for the data and metadata, figure out
        and load the appropriate BLM type & coord mapping.

        Args:
            d_t (str, optional): Due to timber detch size limitations, it
                will be split into chunks of d_t. d_t must be a pd.Timedelta
                format str, e.g. 30M, 1H, 10S, ...
            BLM_var (str, optional): Timber/CALS BLM measurement variable.
            pbar (bool, optional): controls whether to display progessbars.
            mute (bool, optional): if True silences any progress prints.
        """
        # progress bar settings
        PBar.use_tqdm = pbar
        PBar.mute = mute

        # header cache
        self.__header = None
        # metadata folder locations
        self.__dcum_folder = Path(__file__).parent / 'metadata' / 'dcum'
        self.__manual_headers_folder = (Path(__file__).parent / 'metadata' /
                                        'headers')

        self.d_t = d_t
        self.BLM_var = BLM_var

        self._logger = logging.getLogger(__name__)
        self._db = DB

    def clear_header(self):
        """Clears cached headers.
        """
        self._logger.debug('Clearing headers.')
        self.__header = None

    def from_datetimes(self, t1, t2, keep_headers=False, **kwargs):
        """Create a BLMData instance with data for the requested interval.
        Note: the time interval cannot cross fill boundaries.

        Args:
            t1 (Datetime): interval start.
            t2 (Datetime): interval end.
            keep_headers (optional, bool): Controls whether to clear the
                headers before fetching data.
            **kwargs: passed to BLMData.__init__.

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
            # try again with bigger fuzzy_t
            fill1 = fill_from_time(t1, fuzzy_t='24H')
            fill2 = fill_from_time(t2, fuzzy_t='24H')

        # TODO: investigate this ... why doesn't it find ...
        if fill1 is None:
            raise ValueError(f'Unable to find fill for {t1}.')
        if fill2 is None:
            raise ValueError(f'Unable to find fill for {t2}.')
        if fill1 != fill2:
            fills = range(fill1['fillNumber'], fill2['fillNumber'])
        else:
            fills = [fill1['fillNumber']]
        data_dfs = []
        fill_bm_dfs = []
        for fill in fills:
            if len(fills) > 1:
                # case where multiple fills
                fill_data = DB.getLHCFillData(fill)
            else:
                # case when fill1 == fill2
                fill_data = fill1
            fill_bm_df = beammode_to_df(fill_data['beamModes'])
            fill_bm_df_mask = ((fill_bm_df >= t1).any() &
                               (fill_bm_df <= t2).any())
            fill_bm_df = fill_bm_df.loc[:, fill_bm_df_mask]
            fill_bm_df = fill_bm_df.applymap(lambda x: np.where(x < t1, t1,
                                                                x))
            fill_bm_df = fill_bm_df.applymap(lambda x: np.where(x > t2, t2,
                                                                x))
            data_dfs.append(self._fetch_data_bm(fill_bm_df))
            # flip the bm timing info to be consistant with the data's
            # structure
            fill_bm_dfs.append(fill_bm_df.T)

        # concat all the fill data and beam mode timings together and add the
        # fill number index level
        data = pd.concat(data_dfs, keys=fills, names=['fill_number'],
                         sort=False)
        fill_bm_df = pd.concat(fill_bm_dfs, keys=fills, names=['fill_number'],
                               sort=False)

        if 'context' not in kwargs:
            kwargs['context'] = fill_bm_df
        if data is None:
            raise ValueError('No BLM data found.')
        return BLMData(data, meta, **kwargs)

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
                 all beam modes, or a list of beam modes to only request a
                 subset.
            unique_beam_modes (bool, optional): If a fill contains multiple of
                the same beam mode, they will be uniquified, INJPHYS, INJPHYS
                --> INJPHYS, INJPHYS_2. Setting unique_beam_modes to True and
                providing INJPHYS_2 in beam_modes will select the second
                INJPHYS beam mode.
            keep_headers (optional, bool): Controls whether to clear the
                headers before fetching data.
            **kwargs: passed to BLMData.__init__.

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

    def bg_from_INJPROT(self, fill_number):
        """Fetches BLM data of the INJPROT beam mode when there is no beam.

        Args:
            fill_number (int): fill number of the fill of interest.

        Returns:
            BLMData: BLMData instance with the INJPROT background data.
        """
        fills, bm = self._fetch_beam_modes(fill_number=fill_number,
                                           subset=['INJPROT'],
                                           unique_subset=True)

        # TODO: maybe move this to utils ? and reuse it in get_ADT
        def timber_to_df(t_dict, key):
            out = t_dict[key]
            out = pd.DataFrame(np.vstack(out).T, columns=['timestamp', key])
            out['timestamp'] = pd.to_datetime(out['timestamp'],
                                              unit='s',
                                              utc=True).dt\
                .tz_convert('Europe/Zurich')
            out.set_index('timestamp', inplace=True)
            return out

        t_vars = [INTENSITY.format(beam=1), INTENSITY.format(beam=2)]
        out = DB.get(t_vars,
                     bm['INJPROT']['startTime'],
                     bm['INJPROT']['endTime'])
        b1 = timber_to_df(out, INTENSITY.format(beam=1))
        b2 = timber_to_df(out, INTENSITY.format(beam=2))
        intensity = pd.concat([b1, b2], axis=1)
        no_beam = (intensity < 1e14).all(axis=1)
        t1 = intensity[no_beam].index[0]
        if (~no_beam).any():
            t2 = intensity[~no_beam].index[0]
        else:
            t2 = intensity[no_beam].index[-1]
        return self.from_datetimes(t1, t2, keep_headers=True)

    def bg_from_ADT_trigger(self,
                            trigger_t,
                            dt_prior='0S',
                            dt_post='2S',
                            look_back='2H',
                            look_forward='5S',
                            min_bg_dt='20S',
                            max_bg_dt='10min'):
        """Fetches the appropriate background data by looking at the triggers
        of the ADT and figuring out a correct time range, where no triggers
        occured.

        Args:
            trigger_t (Datetime, optional): Time of ADT trigger, if None, will
                take the timestamp of the first
                value.
            dt_prior (str, optional): time delta prior to adt turn on.
            dt_post (str, optional): time delta post previous adt turn off.
            look_back (str, optional): look back from trigger_t when
                fetching adt trigger data.
            look_forward (str, optional): look forward from trigger_t when
                fetching adt trigger data.
            min_bg_dt (str, optional): minimum amount of time where no ADT
                blowup triggers occur.
            max_bg_dt= (str, optional): maximum amount of time where no ADT
                blowup triggers occur, to limit of the amount of timber
                fetches.

        Returns:
            DataFrame: DataFrame containing the background signal.

        Raises:
            ValueError: if unable to find an region in time with no ADT trigger
                respecting the desired constraints.
        """
        if trigger_t is None:
            trigger_t = self.data.index.get_level_values('timestamp')[0]
        else:
            trigger_t = sanitize_t(trigger_t)

        min_bg_dt = pd.Timedelta(min_bg_dt)
        max_bg_dt = pd.Timedelta(max_bg_dt)
        dt_prior = pd.Timedelta(dt_prior)
        dt_post = pd.Timedelta(dt_post)

        joined = get_ADT(trigger_t - pd.Timedelta(look_back),
                         trigger_t + pd.Timedelta(look_forward),
                         include=['trigger'])
        if not (joined == 1).any(axis=None):
            raise ValueError('No ADT triggers within time range.')

        joined.fillna(method='ffill', inplace=True)
        joined.dropna(axis=0, inplace=True)

        # convert rising/falling edges to "square" functions
        shifted = joined.shift(1)
        # hacky way of making the 'edges' connect properly
        shifted.index -= pd.Timedelta('1MS')
        joined = pd.concat([joined, shifted])
        joined.sort_values(by='timestamp', axis='index', inplace=True)
        # reverse joined to avoid useless iterations
        joined = joined[::-1]
        # find plateaus where the ADT was not triggered
        joined_off = (joined == 0).all(axis=1)
        section = None
        for i, g in joined.groupby([(joined_off != joined_off.shift()).cumsum()]):
            data_range = g.index[0] - g.index[-1] - dt_prior - dt_post
            if not g.any().any() and data_range >= min_bg_dt:
                section = g[::-1]
                self._logger.debug(f"Found background {i} plateaus back.")
                break

        if section is None:
            msg = ("Failed to find adequate ADT off plateau. "
                   "Consider relaxing the 'min_bg_dt', 'dt_post' "
                   " and 'dt_prior' constraints, "
                   "or maybe increasing the 'look_back' amount.")
            raise ValueError(msg)

        t1 = section.index[0] + dt_post
        t2 = section.index[-1] - dt_prior

        if t2 - t1 > max_bg_dt:
            t1 = t2 - max_bg_dt

        self._logger.debug(f'Background timestamp t1: {t1}')
        self._logger.debug(f'Background timestamp t2: {t2}')

        return self.from_datetimes(t1, t2, keep_headers=True)

    def iter_from_ADT(self, t1, t2,
                      look_forward='5S',
                      look_back='0S',
                      planes=['H', 'V'],
                      beams=[1, 2],
                      keep_headers=False,
                      yield_background=False,
                      include=['trigger', 'amp', 'length', 'gate'],
                      conditions={},
                      ):
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
            include (list, optional): which ADT information to fetch, must be
                a key of utils.BEAM_META. Will be passed on the the returned
                BLMData's context attribute.
            conditions (dict, optional): dictionnay containing as key an
                element of "include" and value a function returning a bool.
                Example:
                conditions={'amp': lambda x: x > 0.6} will only return data for
                ADT triggers with an excitation amplitude > 0.6. Multiple
                condition will be combined with AND logic.

        Yields:
            BLMData: BLMData instance with data surrounding the ADT
                trigger.

        Raises:
            ValueError: if 'conditions' are badly set, if no ADT triggers are
                found within time range.
        """

        # TODO: figure out if this conditions and include things are the best
        #  way of doing this...
        if not set(conditions.keys()) <= set(include):
            raise ValueError('"condition" keys must be in "include" list.')
        if 'trigger' not in include:
            include.append('trigger')

        if not keep_headers:
            self.clear_header()

        t1 = sanitize_t(t1)
        t2 = sanitize_t(t2)

        joined = get_ADT(t1, t2, planes=planes, beams=beams, include=include)
        joined = joined.fillna(method='ffill')

        if not (joined == 1).any(axis=None):
            # TODO: cleanup the pbar thing ...
            raise ValueError('No ADT triggers within time range.')

        for c in itertools.product(beams, planes):
            # beam token
            c = 'B' + ''.join(map(str, c))
            # work on subset for current beam/plane
            sub_joined = joined.filter(regex=f'.*{c}.*')
            triggers = sub_joined[(sub_joined[f"ADT_{c}_trigger"] == 1)]

            # apply condition filtering
            for v, cond in conditions.items():
                triggers = triggers[cond(triggers[f'ADT_{c}_{v}'])]

            # create iterable
            triggers_iter = triggers.iterrows()
            triggers_iter = PBar(triggers_iter, desc=c, total=len(triggers))

            for t, row in triggers_iter:
                # add a few ease of life keys to the row.
                row['ADT_beam'] = c[1]
                row['ADT_plane'] = c[-1]
                row['ADT_datetime'] = t
                try:
                    BLM_data = self.from_datetimes(t - pd.Timedelta(look_back),
                                                   t + pd.Timedelta(look_forward),
                                                   context=row,
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
        """Gets the metadata (dcum and type) corresponding to the requested timestamp.

        Args:
            t (Datetime): Time of data.

        Returns:
            DataFrame: DataFrame containing blm, position and type.
        """
        dcum_df = files_to_df(self.__dcum_folder)
        meta_time_ind = dcum_df.index.get_loc(t, method='ffill')
        meta_time = dcum_df.index[meta_time_ind]
        self._logger.debug(f'Using BLM metadata from {meta_time}.')
        dcum_file = self.__dcum_folder / dcum_df.loc[meta_time].values[0]
        meta = pd.read_csv(dcum_file, index_col=0)
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
        blms = self.fetch_meta(t).index.tolist()
        blms = [b for b in blms if b.split('.')[0] not in ['BLMTS', 'BLMMI',
                                                           'BLMES', 'BLMDS',
                                                           'BLMCK', 'BLMAS',
                                                           'BLMCD']]
        return blms

    def fetch_manual_header(self, t):
        """Fetch the fill's column names from manual header files.

        Args:
            t (DateTime): Time of data.

        Returns:
            list: list of columns containing the name of BLMs.
        """
        # TODO: this order of these headers can be slightly off...
        manual_header_df = files_to_df(self.__manual_headers_folder)
        manual_header_ind = manual_header_df.index.get_loc(t, method='ffill')
        manual_header_file = (self.__manual_headers_folder /
                              manual_header_df.iloc[manual_header_ind].values[0])
        with open(manual_header_file) as fp:
            blms = [line.rstrip() for line in fp]

        return blms

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

    def _fetch_data_t(self, t1, t2):
        """Fetch the data for one time chunk.

        Args:
            t1 (Datetime): Start time of the data fetch.
            t2 (Datetime): End time of the data fetch.

        Returns:
            DataFrame: DataFrame containing the BLM data.
        """
        header_fetchers = [self.fetch_timber_header,
                           self.fetch_manual_header,
                           self.fetch_logging_header]

        data = self._db.get(self.BLM_var, t1, t2)[self.BLM_var]
        if data[1].size == 0:
            return

        if self.__header is None:
            for fetcher in header_fetchers:
                columns = fetcher(t1)
                self._logger.debug(f'Header from self.{fetcher.__name__}: {len(columns)}.')
                # self._logger.debug(f'Header from self.{fetcher.__name__}:\n{columns}')
                self._logger.debug(f'Number of columns in data: {data[1].shape[1]}.')
                if len(columns) == data[1].shape[1]:
                    self._logger.debug(f'Using header from self.{fetcher.__name__}.')
                    self.__header = columns
                    break

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
                columns and "startTime", "endTime" as
                index.

        Returns:
            DataFrame: MultiIndex DataFrame containing the BLM data for the
               beam modes.
        """
        # TODO: cleanup the pbar thing ...
        data = []
        bm_iter = beam_modes.columns
        bm_iter = PBar(beam_modes.columns, desc='Beam mode')
        for mode in bm_iter:
            bm_iter.set_description(f'Beam mode {mode}')

            t_chunks = list(self._date_chunk(beam_modes[mode]['startTime'],
                                             beam_modes[mode]['endTime'],
                                             pd.Timedelta(self.d_t)))
            d = []
            t_iter = zip(t_chunks, t_chunks[1:])
            t_iter = PBar(t_iter, total=len(t_chunks) - 1)
            for t1, t2 in t_iter:
                t_iter.set_description(t_chunks[0].strftime("%m-%d %H:%M:%S") +
                                       ' ▶' +
                                       t_chunks[-1].strftime("%m-%d %H:%M:%S"))
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
