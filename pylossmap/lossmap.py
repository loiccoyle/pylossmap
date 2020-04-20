# import copy
import warnings
import logging
import pandas as pd

from functools import partial

from .utils import DB
from .utils import BEAM_META
from .utils import coll_meta
from .utils import angle_convert
from .plotting import plot_loss_map
from .blm_filters import Filters

# index str contains userwarning ignore
warnings.filterwarnings("ignore", 'This pattern has match groups')


class LossMap(Filters):
    def __init__(self,
                 data,
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
        self.df = data
        self._background = None
        self.context = context

        # Dynamically add beam meta fetching methods
        for k, (v, series) in BEAM_META.items():

            meth = partial(self.fetch_var, v)
            name = f'fetch_{k}'
            meth.__name__ = name
            meth.__doc__ = (f"Fetches the value {k} from timber closest in "
                            "time to the datetime attribute.\n"
                            "Args:\n"
                            "\tv (str, optional): Timber variable.\n"
                            "\treturn_raw (bool, optional): if True, returns "
                            "the timestamps along with the data.\n"
                            "\n"
                            "Returns:\n"
                            "\tfloat or tuple: data point, if return_raw is "
                            "True, returns tuple (timestamp, data).")
            setattr(self, name, meth)

    def fetch_var(self, v, return_raw=False, **kwargs):
        """Fetches the value of a timber variable closest in time to the
        datetime attribute.

        Args:
            v (str, optional): Timber variable.
            return_raw (bool, optional): if True, returns the timestamps along
                with the data.

        Returns:
            float or tuple: data point, if return_raw is True, returns tuple
                (timestamp, data).
        """
        self._check_datetime()
        try:
            v = v.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Provide {e} kwarg.")
        out = DB.get(v, self.datetime)[v]
        if not return_raw:
            out = out[1][0]
        return out

    @property
    def meta(self):
        return self.df[self._meta_cols]

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
        out = LossMap(data=self.df,
                      datetime=self.datetime,
                      context=self.context)
        if self._background is not None:
            out.set_background(self._background)
        return out
        # return copy.deepcopy(self)

    def filter(self, reg, mask=False):
        """Applies a regexp filter to the BLM names a returns a filters LossMap
        instance.

        Args:
            reg (str): regexp string.
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap instance.

        Returns:
            LossMap or boolean array: LossMap instance or boolean mask array
                containing the BLMs which matched the regex string.
        """

        if not mask:
            ret = self.copy()
            ret.df = ret.df.filter(regex=reg, axis='index')
        else:
            ret = self.df.index.str.contains(reg, regex=True)
        return ret

    def _blm_list_filter(self, blm_list):
        ret = self.copy()
        ret.df = ret.df.loc[blm_list]
        return ret

    def normalize(self, wrt='max'):
        """Normalizes the loss map data.

        Args:
            wrt (str, optional): Either 'max' or a BLM name.

        Returns:
            LossMap: LossMap instance normalized.
        """
        if wrt == 'max':
            normalizer = self.df['data'].max()
        elif wrt in self.df.keys():
            normalizer = self.df['data'][wrt]
        ret = self.copy()
        ret.df['data'] /= normalizer
        return ret

    def clean_background(self):
        """Substracts the background from the data.

        Returns:
            LossMap: LossMap instance with cleaned data.
        """
        if self._background is None:
            raise ValueError('background not set, use self.set_background.')
        ret = self.copy()
        ret.df['data'] = ret.df['data'] - ret._background.df['data']
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
        b = self.beam(beam).df['data']
        b = b.sum()
        return b/self.df['data'].sum()

    def cleaning(self, IR=7):
        '''Cleaning efficiency ?
        TODO : make sure this is correct.

        Args:
            IR (int): either IR 5 or 7.

        Returns:
            float: cleaning efficiency
        '''
        return (self.IR(IR).DS().df['data'].max() /
                self.IR(IR).TCP().df['data'].max())

    def __getitem__(self, key):
        ret = self.copy()
        ret.df = ret.df.__getitem__(key)
        return ret

    def __setitem__(self, key, value):
        self.df.__setitem__(key, value)

    # def __add__(self, other):
    #     ret = self.copy()
    #     return ret.data['data'] + other.data['data']

    # def __sub__(self, other):
    #     ret = self.copy()
    #     return ret.data['data'] - other.data['data']

    def __repr__(self):
        bg_str = ""
        if self._background is not None:
            bg_str = f"\n\tbackground:\n{self._background.df.__repr__()}"

        return f"LossMap:\n\
\tdf:\n{self.df.__repr__()}" + bg_str

    def plot(self, data=None, **kwargs):
        if data is None:
            data = self.df['data']
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

        weighted = self.df['data'] * self.df[f'{plane}_weight']
        return weighted.sum() / self.df[f'{plane}_weight'].sum()

    def plane(self, plane):
        # TODO: add SKEW?
        plane_angle = {'H': 1.571,
                       'V': 0}
        if plane not in plane_angle.keys():
            raise ValueError('"plane" must be either "H" or "V".')

        ret = self.copy()
        ret.df[ret.df['coll_angle'] == plane_angle[plane]]
        return ret
