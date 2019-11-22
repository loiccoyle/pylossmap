import copy
import logging
import pandas as pd

from .utils import DB
from .utils import BEAM_META
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
            context (optional): additional context information.
        '''

        self._logger = logging.getLogger(__name__)
        self._meta_cols = ['type', 'dcum']
        self.datetime = datetime
        self.data = data
        self._background = None
        self.context = context

    @property
    def meta(self):
        return self.data[self._meta_cols]

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

    def set_background(self, LM_bg):
        '''Links the provided background LossMap to this one.

        Args:
            LM_bg (LossMap): Background LossMap

        Raises:
            ValueError: If the provided background is not a LossMap instance.
        '''
        if not isinstance(LM_bg, LossMap):
            raise ValueError('"LM_bg" must be a LossMap instance.')
        self._background = LM_bg.copy()

    def get_background(self):
        '''Returns the linked background LossMap

        Returns:
            LossMap: the linked LossMap instance.
        '''
        return self._background

    def _check_datetime(self):
        if self.datetime is None:
            raise ValueError("'datetime' attribute must be set to be able to "
                             "fetch Timber data")

    def copy(self):
        """Creates a copy of the current instance.

        Returns:
            LossMap: Copied LossMap instance.
        """
        return copy.deepcopy(self)

    def filter(self, reg):
        """Applies a regexp filter to the BLM names a returns a filters LossMap
        instance.

        Args:
            reg (str): regexp string.

        Returns:
            LossMap: LossMap instance containing the BLMs which matched the
            regex string.
        """
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

    def clean_background(self):
        """Substracts the background from the data.

        Returns:
            LossMap: LossMap instance with cleaned data.
        """
        if self._background is None:
            raise ValueError('background not set, use self.set_background.')
        ret = self.copy()
        ret.data['data'] = ret.data['data'] - ret._background.data['data']
        return ret

    def DS(self):
        """Selects the BLMs in the dispersion supperssor region.

        Returns:
            LossMap: LossMap instance containing the dispersion suppressors
            BLMs.
        """
        return self.filter(rf'BLMQ[IE]\.(0[7-9]|10|11)[RL][37]')

    def IR(self, *IRs):
        """Filters the BLMs based on the IR(s).

        Args:
            *IRs (int): IR(s) of interests.

        Returns:
            LossMap: LossMap instance with the filtered IR(s).
        """
        IR = self._sanitize_inp(IRs,
                                check={1, 2, 3, 4, 5, 6, 7, 8})
        return self.filter(rf'\.\d\d[LR]({IR})')

    def TCP(self, HVS=False):
        """Selects only the TCP BLMs.

        Returns:
            LossMap: LossMap instance containing the TCP BLMs.
        """
        if HVS:
            pattern = r'BLMTI.*TCP\.'
        else:
            pattern = r'TCP\.'
        return self.filter(pattern)

    def TCS(self):
        """Selects only the TCS BLMs.

        Returns:
            LossMap: LossMap instance containing the TCS BLMs.
        """
        return self.filter(r'TCS[GP][M]?\.')

    def TCL(self):
        """Selects only the TCL BLMs.

        Returns:
            LossMap: LossMap instance containing the TCS BLMs.
        """
        return self.filter(r'TCL[A]?\.')

    def TCTP(self):
        """Selects only the TCTP BLMs.

        Returns:
            LossMap: LossMap instance containing the TCTP BLMs.
        """
        return self.filter(r'TCTP[HV]\.')

    def TCLI(self):
        """Selects only the TCLI BLMs.

        Returns:
            LossMap: LossMap instance containing the TCLI BLMs.
        """
        return self.filter(r'TCLI[AB]\.')

    def side(self, RL):
        """Filters the BLMs based on their side.

        Args:
            RL (str): Either "R" or "L" or "RL".

        Returns:
            LossMap: LossMap instance with the filtered BLMs.
        """
        return self.filter(rf'\.\d\d[{RL}][1-8]')

    def cell(self, *cells):
        """Filters the BLMs based on their cell number(s).

        Args:
            *cells (int): cells of interest.

        Returns:
            LossMap: LossMap instance with the filtered cells.
        """

        def pad(x):
            return f'{x:02}' if x < 10 else str(x)

        cells = self._sanitize_inp(cells,
                                   prepare=pad)
        return self.filter(rf'\.({cells})[RL][1-8]')

    def beam(self, *beams):
        """Filters the BLMs based on the beam(s).

        Args:
            *beam (int): Beams of interest, subset of {0,1,2}.

        Returns:
            LossMap: LossMap instance with the filtered beam(s).
        """
        beam = self._sanitize_inp(beams,
                                  check={0, 1, 2},
                                  prepare=lambda x: str(int(x)))
        return self.filter(rf'B({beam})')

    def type(self, types):
        '''Gets the BLM for the requested blm types.

        Args:
            types (list/str): string or list of the types of interest.

        Returns:
            LossMap: LossMap instance with the desired BLM types.
        '''
        if not isinstance(types, list):
            types = [types]

        ret = self.copy()
        ret.data = self.data[self.data['type'].isin(types)]
        return ret

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
            float: requested beam summed/total losses.
        """
        b = self.beam(beam).data['data']
        b = b.sum()
        return b/self.data['data'].sum()

    def cleaning(self, IR=7):
        '''Cleaning efficiency ?
        TODO : make sure this is correct.

        Args:
            IR (int): either IR 5 or 7.

        Returns:
            float: cleaning efficiency
        '''
        return (self.IR(IR).DS().data['data'].max() /
                self.IR(IR).TCP().data['data'].max())

    def __getitem__(self, key):
        ret = self.copy()
        ret.data = ret.data.__getitem__(key)
        return ret

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    # def __add__(self, other):
    #     ret = self.copy()
    #     return ret.data['data'] + other.data['data']

    # def __sub__(self, other):
    #     ret = self.copy()
    #     return ret.data['data'] - other.data['data']

    def __repr__(self):
        bg_str = ""
        if self._background is not None:
            bg_str = f"\n\tbackground:\n{self._background.data.__repr__()}"

        return f"LossMap:\n\
\tdata:\n{self.data.__repr__()}" + bg_str

    def plot(self, data=None, **kwargs):
        if data is None:
            data = self.data['data']
        return plot_loss_map(data=data,
                             meta=self.meta,
                             **kwargs)


# Dynamically add beam meta fetching methods
for k, (v, series) in BEAM_META.items():

    def fetch(self, v=v, return_raw=False, **kwargs):
        self._check_datetime()
        v = v.format(**kwargs)
        out = DB.get(v, self.datetime)[v]
        if not return_raw:
            out = out[1][0]
        return out
    name = f'get_{k}'
    fetch.__name__ = name
    fetch.__doc__ = (f"Gets the {k} value from timber closest to the datetime"
                     " attribute.\n"
                     "Args:\n"
                     "\tv (str, optional): Timber varibale.\n"
                     "\treturn_raw (bool, optional): if True, returns "
                     "the timestamps along with the data.\n"
                     "\n"
                     "Returns:\n"
                     "\tfloat or tuple: data point, if return_raw is "
                     "True, returns tuple (timestamp, data).")
    setattr(LossMap, name, fetch)


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
        """Restricts BLMs to the collimator BLMs and adds adds the collimator
        angle read from file, to the data attribute.

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
