import copy
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .blm_filters import Filters
from .lossmap import LossMap
from .plotting import plot_waterfall
from .timber_vars import PRIMARY_BLM_7
from .type_specs import PathSpec
from .utils import BEAM_META, DB, row_from_time, sanitize_t


class BLMData(Filters):
    @classmethod
    def load(cls, file_path: PathSpec, **kwargs) -> "BLMData":
        """Load the data from a hdf file and create a BLMData instance.

        Args:
            file_path: Path to hdf file from which to load the data.

        Returns:
            BLMData instance with the loaded data.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        file_path = Path(file_path).with_suffix(".h5")
        if not file_path.is_file():
            raise FileNotFoundError(f"File {file_path} not found.")

        data = pd.read_hdf(file_path, "data")
        meta = pd.read_hdf(file_path, "meta")
        header = pd.read_hdf(file_path, "header")
        header = header[0].tolist()
        # # read real columns from csv file & replace fake columns
        # with open(file_path.with_suffix('.csv'), 'r') as fp:
        #     columns = fp.readlines()
        data.columns = [c.rstrip() for c in header]
        return cls(data, meta, **kwargs)

    def __init__(
        self,
        data: pd.DataFrame,
        meta: pd.DataFrame,
        BLM_filter: Union[str, List[str]] = r"BLM[Q|B|A|T|2|E]I.*",
        context: Optional[Any] = None,
    ):
        """This class handles the parsing/preprocessing & plotting of the BLM data.

        Args:
            data: MultiIndex DataFrame containing the BLM
                measurements, for the various beam modes & query chunks.
            meta: DataFrame containing the BLM metadata.
            BLM_filter: regex str of list of regex strs,
                BLMs of interest.
            context: additional info.
        """
        self._logger = logging.getLogger(__name__)
        self.context = context

        self.df = data
        if BLM_filter is not None:
            if not isinstance(BLM_filter, list):
                BLM_filter = [BLM_filter]
            for f in BLM_filter:
                self.df = self.df.filter(regex=f, axis="columns")
        self.meta = self._get_metadata(meta)

        # Dynamically add beam meta fetching methods
        for k, (v, series) in BEAM_META.items():
            meth = partial(self.fetch_var, v, timeseries=series)
            name = f"fetch_{k}"
            meth.__name__ = name
            meth.__doc__ = (
                f"Gets {k} data from timber for the current "
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
                "containing timestamp and data arrays."
            )
            setattr(self, name, meth)

    def fetch_var(
        self,
        v: str,
        timeseries: bool = True,
        return_raw: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Gets provided timber variable data from timber for the current time range.

        Args:
            v: Timber variable.
            timeseries: whether Timber variable returns a timeseries.
            return_raw: if True, returns the timestamps along with the data.
            **kwargs: if `v` is a formattable string, `v.format(**kwargs)`

        Returns:
            DataFrame or tuple: Dataframe with timestamp and data. If return_raw
                is True, a tuple containing timestamp and data arrays.
        """
        t1 = self.df.index.get_level_values("timestamp")[0]
        t2 = None
        if timeseries:
            t2 = self.df.index.get_level_values("timestamp")[-1]
        try:
            v = v.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Provide {e} kwarg.")
        out = DB.get(v, t1, t2)[v]
        if not return_raw:
            out = pd.DataFrame(np.vstack(out).T, columns=["timestamp", v])
            out["timestamp"] = pd.to_datetime(
                out["timestamp"], unit="s", utc=True
            ).dt.tz_convert("Europe/Zurich")
            out = out.set_index("timestamp")
        return out

    def copy(self) -> "BLMData":
        """Creates a copy of the current instance.

        Returns:
            BLMData: Copied BLMData instance.
        """
        # TODO: check if this fails, _thread.RLock...
        return copy.deepcopy(self)

    def filter(self, reg: str, mask: bool = False) -> Union["BLMData", np.ndarray]:
        """Applies a regexp filter to the BLM names a returns a filtered BLMData
        instance.

        Args:
            reg: regexp string.
            mask: if True will return a boolean mask array, otherwise will return
                a filtered LossMap instance.

        Returns:
            BLMData instance or boolean mask array containing the BLMs which matched
                the regex string.
        """

        if not mask:
            ret = self.copy()
            ret.df = ret.df.filter(regex=reg, axis="columns")
        else:
            ret = self.df.columns.str.contains(reg, regex=True)
        return ret

    def _blm_list_filter(self, blm_list: List[str]) -> "BLMData":
        ret = self.copy()
        blm_list_common = list(set(blm_list) & set(ret.df.columns))
        ret.df = ret.df[blm_list_common]
        return ret

    def find_max(self, BLM_max: Optional[List[str]] = None) -> pd.DataFrame:
        """Finds the max timestamp and chunk in which the max occured.

        Args:
            BLM_max: List of BLMs, defaults to the primary
                blms in IR 7.

        Returns:
            DataFrame containing a tuple: (mode, datetime).
        """
        if BLM_max is None:
            BLM_max = PRIMARY_BLM_7[1] + PRIMARY_BLM_7[2]

        maxes = self.df.groupby("mode")[BLM_max].idxmax()
        # tuples are (beam_mode, timestamp)
        # only keep timestamp
        return maxes.applymap(lambda x: x[1])

    def iter_max(self, BLM_max: Optional[List[str]] = None):
        """Creates an generator of ((mode, datetime), BLM_max), where mode and
        datetime correspond to the max value of BLM_max.

        Args:
            BLM_max: List of BLMs, defaults to the primary
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
        maxes = pd.DataFrame(iterable, columns=["mode", "timestamp", "blm"]).groupby(
            ["mode", "timestamp"], sort=False
        )
        return ((r[0], r[1]["blm"].tolist()) for r in maxes)

    def _get_metadata(self, meta: pd.DataFrame) -> pd.DataFrame:
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
        without_meta_df = pd.DataFrame({"blm": without_meta}).set_index("blm")
        without_meta_df["type"] = "other"
        without_meta_df["dcum"] = None
        return (
            pd.concat([with_meta_df, without_meta_df], sort=False)
            .sort_values("dcum")
            .astype({"dcum": "int64", "type": "str"})
        )

    def loss_map(
        self,
        datetime: Optional[pd.Timestamp] = None,
        row: Optional[pd.Series] = None,
        context: Optional[Any] = None,
        background: Optional[pd.Series] = None,
        **kwargs,
    ) -> LossMap:
        """Creates a LossMap instance.

        Args:
            datetime: If provided, is used to find the row in the data closest
                to datetime.
            row: Row of data for which to create the LossMap instance.
            context: if None, will use self.context.
            background: if provided will create a LossMap instance for the background
                and set is as the background of the returned LossMap instance.
            **kwargs: passed to LossMap.__init__.

        Returns:
            LossMap instance of the desired data.

        Raises:
            ValueError: If neither `datetime` nor `row` is provided.
        """
        if datetime is None and row is None:
            raise ValueError('Provide either "datetime", or "row".')
        if row is None:
            row = row_from_time(self.df, datetime, flatten=True, method="nearest")
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
        data.columns = ["data", "dcum", "type"]
        LM = LossMap(data=data, datetime=datetime, context=context, **kwargs)

        if background is not None:
            background = pd.concat([background, self.meta], axis=1, sort=False)
            background.columns = ["data", "dcum", "type"]
            background = LossMap(data=background, context=context)
            LM.set_background(background)
        return LM

    def save(self, file_path: PathSpec):
        """Save the DataFrames to hdf file/keys.

        Keys:
            data: contains the blm data.
            meta: contains the blm metadata.
            header: contains the blm header, i.e. the BLM names.

        Args:
            file_path: Path to hdf file in which to save the DataFrames.

        Raises:
            OSError: If file already exists.
        """
        file_path = Path(file_path).with_suffix(".h5")
        if file_path.is_file():
            raise OSError(f"File {file_path} already exists.")

        # remove BLM names from columns due to hdf header size limitation.
        # save_data = self.df.copy()
        header = self.df.columns
        try:
            # replace real header with numbers to not have problem saving large
            # header to hdf
            self.df.columns = range(self.df.shape[1])
            self.df.to_hdf(file_path, key="data", format="table")
            self.meta.to_hdf(file_path, key="meta", format="table", append=True)
            pd.DataFrame(header).to_hdf(
                file_path, key="header", format="table", append=True
            )
        finally:
            # put real header back
            self.df.columns = header
        # # write columns real columns name in separate file.
        # with open(file_path.with_suffix('.csv'), 'w') as fp:
        #     fp.write('\n'.join(self.df.columns))

    def plot(
        self, data: Optional[pd.DataFrame] = None, title: Optional[str] = None, **kwargs
    ) -> Union[Tuple[plt.Figure, plt.Axes], Dict[str, Tuple[plt.Figure, plt.Axes]]]:
        """Plots a waterfall plot of the data. Note, will produce multiple figures
        if data contains a mode index.

        Args:
            data: DataFrame containing the BLM data.
            title: figure title, '{mode}' gets replaced with the beam mode.
            **kwargs: passed to plotting.plot_waterfall

        Returns:
            If the data contains multiple beam modes, returns a dictionary of beam
                mode and Figure, Axes. Otherwise just the Figure and Axes.
        """
        # TODO: fix this format mode thing
        if data is None:
            data = self.df
        if "mode" in data.index.names:
            out = {}
            for mode, d in data.groupby("mode"):
                if title is not None:
                    kwargs["title"] = title.format(mode=mode)
                else:
                    kwargs["title"] = mode

                out[mode] = plot_waterfall(data=d, meta=self.meta, **kwargs)
            return out
        else:
            return plot_waterfall(data=data, meta=self.meta, **kwargs)

    def __getitem__(self, key: Union[int, float, str]) -> LossMap:
        """
        Args:
            key: if int then the corresponding data row is used to create a LossMap
                instance. If float or str then assumed unix time or pd.to_datetime
                compatible str.

        Returns:
            LossMap: LossMap instance with the desired BLM data.
        """
        if isinstance(key, int) and key < self.df.size:
            # get row time from data index.
            time = self.df.index.get_level_values("timestamp")[key]
        else:
            # assume epoch or to_datetime str.
            time = sanitize_t(key)
        return self.loss_map(datetime=time)

    def __repr__(self) -> str:
        out = "df:\n" + self.df.__repr__() + "\nmeta:\n" + self.meta.__repr__()
        if self.context is not None:
            out += "\ncontext:\n" + self.context.__repr__()
        return out
