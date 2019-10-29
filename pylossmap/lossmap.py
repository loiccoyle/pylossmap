import copy
import logging
import pandas as pd

from .utils import coll_meta
from .utils import angle_convert
from .plotting import plot_loss_map


class LossMap:
    def __init__(self,
                 data,
                 background=None,
                 datetime=None,
                 context=None,
                 ):
        '''Handles the processing of the LossMap data.

        Args:
            data (DataFrame): Dataframe contraining BLM data, dcum, type.
            background (DataFrame, optional): BLM background signal.
            datetime (Datetime, optional): Datetime of the data.
        '''

        self._logger = logging.getLogger(__name__)
        self._meta_cols = ['type', 'dcum']
        self.datetime = datetime
        self.data = data
        self.background = background
        self.context = context

    @property
    def meta(self):
        return self.data[self._meta_cols]

    def copy(self):
        """Creates a copy of the current instance.

        Returns:
            LossMap: Copied LossMap
        """
        return copy.deepcopy(self)

    @staticmethod
    def _sanitize_inp(inp, prepare=None, check=None):
        if check is not None:
            if not set(inp) <= check:
                raise ValueError(f"Input must be subset of {check}.")

        if callable(prepare):
            inp = map(prepare, inp)
        else:
            inp = map(str, inp)
        return '|'.join(inp)

    def filter(self, reg):
        ret = self.copy()
        ret.data = ret.data.filter(regex=reg, axis='index')
        return ret

    def normalize(self, wrt='max'):
        """Normalizes the loss map data.

        Args:
            wrt (str, optional): Either 'max' or a BLM name.

        Returns:
            LossMap: LossMap instance normalized.
        """
        if wrt == 'max':
            normalizer = self.data['data'].max()
        elif wrt in self.data.keys():
            normalizer = self.data['data'][wrt]
        ret = self.copy()
        ret.data['data'] /= normalizer
        return ret

    def clean_bg(self):
        """Removes the background from the data.

        Returns:
            LossMap: LossMap instance with cleaned data.
        """
        ret = self.copy()
        if self.background is not None:
            ret.data['data'] = ret.data['data'] - ret.background.mean()
        return ret

    def DS(self):
        """Gets the dispersion suppressor data.

        Returns:
            LossMap: LossMap instance of dispersion suppressors.
        """
        return self.filter(rf'BLMQ[IE]\.(0[7-9]|10)[RL][37]')

    def IR(self, *IRs):
        """Gets the data of a given IR.

        Args:
            *IRs (int): IR(s) of interests.

        Returns:
            LossMap: LossMap instance of the requiested IR.
        """
        IR = self._sanitize_inp(IRs,
                                check={1, 2, 3, 4, 5, 6, 7, 8})
        return self.filter(rf'\.\d\d[LR]({IR})')

    def TCP(self, HVS=False):
        """Gets the data for the TCPs from data.

        Returns:
            LossMap: LossMap instance of the TCPs.
        """
        if HVS:
            pattern = r'BLMTI.*TCP\.'
        else:
            pattern = r'TCP\.'
        return self.filter(pattern)

    def TCS(self):
        return self.filter(r'TCS[GP][M]?\.')

    def TCL(self):
        return self.filter(r'TCL[A]?\.')

    def TCTP(self):
        return self.filter(r'TCTP[HV]\.')

    def TCLI(self):
        return self.filter(r'TCLI[AB]\.')

    def side(self, RL):
        return self.filter(rf'\.\d\d[{RL}][1-8]')

    def cell(self, *cells):
        """Gets the data for requested cells.

        Args:
            *cells (int): cells of interest.

        Returns:
            LossMap: LossMap of the desired cells.
        """
        pad = lambda x: f'{x:02}' if x < 10 else str(x)
        cells = self._sanitize_inp(cells, prepare=pad)
        return self.filter(rf'\.({cells})[RL][1-8]')

    def beam(self, *beams):
        """Gets the data for a specific beam.

        Args:
            *beam (int): Beams of interest, subset of {0,1,2}.

        Returns:
            LossMap: LossMap of the desired beam.
        """
        beam = self._sanitize_inp(beams,
                                  check={0, 1, 2},
                                  prepare=lambda x: str(int(x)))
        return self.filter(rf'B({beam})')

    def type(self, types):
        if not isinstance(types, list):
            types = [types]

        ret = self.copy()
        ret.data = self.data[self.data['type'].isin(types)]
        return ret

    def __getitem__(self, key):
        ret = self.copy()
        return ret.data.__getitem__(key)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)
    # def TCP_plane(self, beam='auto'):
    #     if beam not in ['auto', 1, 2]:
    #         raise ValueError('"beam" must be either "auto", 1, or 2.')

    #     if beam == 'auto':
    #         ratios = [self.beam_ratio(beam) for beam in [1, 2]]
    #         beam = [1, 2][ratios.index(max(ratios))]
    #         self._logger.info(f'Identifying plane for B{beam}.')

    #     TCP_orient = {'D6[RL]7': 'V',
    #                   'C6[RL]7': 'H',
    #                   'B6[RL]7': 'S'}

    #     LM_beam = self.beam(beam)
    #     self._logger.info(LM_beam.IR(7).filter(r'BLMTI.*TCP\.').data['data'])
    #     min_loss_blm = LM_beam.IR(7).filter(r'BLMTI.*TCP\.').data['data'].idxmin()
    #     for k, v in TCP_orient.items():
    #         if re.search(k, min_loss_blm):
    #             return v

    def beam_ratio(self, beam):
        """Beam loss ratio.

        Args:
            beam (int): requested beam.

        Returns:
            float: requsted beam summed/total losses.
        """
        b = self.beam(beam).data['data']
        b = b.sum()
        return b/self.data['data'].sum()

    def cleaning(self, IR=7):
        return self.IR(IR).DS().data['data'].max() / self.IR(IR).TCP().data['data'].max()

    def __repr__(self):
        return f"LossMap:\n\
\tdata:\n{self.data.__repr__()}\n\
\tbackground:\n{self.background.__repr__()}"

    # def restore(self):
    #     self.data = self._data_bk.copy()
    #     return self

    def plot(self, data=None, **kwargs):
        if data is None:
            data = self.data['data']
        return plot_loss_map(data=data,
                             meta=self.meta,
                             **kwargs)


class CollLossMap(LossMap):

    @classmethod
    def from_loss_map(cls, loss_map, coll_df=None):
        '''Imports and converts a LossMap instance.
        '''
        return cls(loss_map.data,
                   background=loss_map.background,
                   datetime=loss_map.datetime,
                   context=loss_map.context,
                   coll_df=coll_df)

    def __init__(self, data, coll_df=None, **kwargs):
        """Collimation loss map, this class adds the collimator angle read from
        file, to the data attribute.

        Args:
            data (DataFrame): Dataframe contraining BLM data, dcum, type.
            coll_df (DataFrame, optional): DataFrame containing collimator
            info.
            **kwargs: passed to LossMap.__init__.
        """
        if coll_df is None:
            coll_df = coll_meta(augment_b2=True)

        out = []
        for blm in coll_df.index:
            tmp = data.loc[data.index.str.contains(blm)].copy()
            if tmp.empty:
                continue
            tmp['coll_angle'] = coll_df.loc[blm, 'angle']
            out.append(tmp)
        data = pd.concat(out)

        for plane in ['H', 'V']:
            if plane == 'V':
                data[f'{plane}_weight'] = data['coll_angle'].map(angle_convert)
            elif plane == 'H':
                data[f'{plane}_weight'] = 1 - data['coll_angle'].map(angle_convert)

        super().__init__(data, **kwargs)
        self._meta_cols = ['type',
                           'dcum',
                           'coll_angle',
                           'H_weight',
                           'V_weight']

    def plane_w_avg(self, plane):
        """Calculate the weighted average losses for a given plane.

        Args:
            plane (str): "H" or "V"

        Returns:
            float: weighted loss average.

        Raises:
            ValueError: if plane is not "H" or "V".
        """
        if plane not in ['H', 'V']:
            raise ValueError('"plane" must be either "H" or "V".')

        weighted = self.data['data'] * self.data[f'{plane}_weight']
        return weighted.sum() / self.data[f'{plane}_weight'].sum()

    def plane(self, plane):
        # TODO: add SKEW?
        plane_angle = {'H': 1.571,
                       'V': 0}
        if plane not in plane_angle.keys():
            raise ValueError('"plane" must be either "H" or "V".')

        ret = self.copy()
        ret.data[ret.data['coll_angle'] == plane_angle[plane]]
        return ret
