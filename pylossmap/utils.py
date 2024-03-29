from typing import Iterable, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import pytimber

from .type_specs import PathSpec, TimeSpec

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


from pathlib import Path

from . import timber_vars

# TODO: Figure out how to get the accelerator mode.
#                              Timber var,               timeseries
BEAM_META = {
    "intensity": (timber_vars.INTENSITY, True),
    "filling_scheme": (timber_vars.FILL_SCHEME, False),
    "number_bunches": (timber_vars.BUNCH_NUM, False),
    "energy": (timber_vars.ENERGY, True),
    # 'amode':          (timber_vars.ACCEL_MODE,  False),
    # 'bmode':          (timber_vars.BEAM_MODE,   False),
}

ADT_META = {
    "amp": timber_vars.ADT_AMP,
    "mode": timber_vars.ADT_MODE,
    "signal": timber_vars.ADT_SIGNAL,
    "gate": timber_vars.ADT_GATE,
    "length": timber_vars.ADT_LENGTH,
    "trigger": timber_vars.ADT_TRIGGER,
}


def get_timber_db(*args, **kwargs):
    """Fetches a pytimber DB insstance."""
    global DB
    if DB is None:
        try:
            DB = pytimber.LoggingDB(*args, **kwargs)
        except (AttributeError, TypeError) as e:
            print(e)
            DB = None
    return DB


DB = None
DB = get_timber_db()  # pytimber.LoggingDB


def uniquify(iterable: Iterable[str]) -> Iterator[str]:
    """Makes the entries in a list unique.

    Args:
        iterable: list to uniquify, duplicates will have "_{number}" added to them.

    Yields:
        uniquified element of iterable
    """
    seen = set()
    for item in iterable:
        fudge = 1
        newitem = item
        while newitem in seen:
            fudge += 1
            newitem = "{}_{}".format(item, fudge)
        yield newitem
        seen.add(newitem)


def to_datetime(ts: Union[int, str]) -> pd.Timestamp:
    """Epoch time to datetime.

    Args:
        ts (int/float): Epoch or Unix time.

    Returns:
        pd.Timestamp instance in Europe/Zurich timezone.
    """
    return pd.to_datetime(ts, unit="s", utc=True).tz_convert("Europe/Zurich")


def fill_from_time(t: TimeSpec, fuzzy_t: str = "12H") -> dict:
    """Gets the machine fill of a timestamp.

    Args:
        t: Epoch/Unix time or timestamp string.
        fuzzy_t: pd.Timedelta format string. Controls the look back and look forward
            around "t" when looking for the fill.

    Returns:
        dict containing the start/end time of the fill and with beam mode info.

    Raises:
        ValueError: if not fill is found.
    """
    t = sanitize_t(t)
    fuzzy_t = pd.Timedelta(fuzzy_t)

    fills = []
    i = 1
    while len(fills) < 2:
        fills = DB.getLHCFillsByTime(t - i * fuzzy_t, t + i * fuzzy_t)
        i += 1

    for fill in fills:
        if to_datetime(fill["startTime"]) <= t and t <= to_datetime(fill["endTime"]):
            return fill
    raise ValueError("Fill not found.")


def beammode_from_time(t: TimeSpec, fill: Optional[dict] = None, **kwargs) -> dict:
    """gets the beam mode at a given timestamp.

    Args:
        t: Epoch/Unix time or timestamp string.
        fill: If provided will skip fetching for the fill and use the provided fill.
        **kwargs: passed to utils.fill_from_time.

    Returns:
        dict: if beam mode if found: beam mode dict, otherwise the fill dict.
    """
    t = sanitize_t(t)
    if fill is None:
        fill = fill_from_time(t, **kwargs)

    for bm in fill["beamModes"]:
        if t >= to_datetime(bm["startTime"]) and t <= to_datetime(bm["endTime"]):
            return bm
    return fill


def beammode_to_df(
    beam_mode, subset: str = "all", unique_subset: bool = False
) -> pd.DataFrame:
    # put beam mode timestamps into dataframe
    beam_mode = pd.DataFrame(beam_mode)
    beam_mode = beam_mode.set_index("mode").T
    beam_mode = beam_mode.applymap(to_datetime)

    if not unique_subset:
        if subset != "all":
            beam_mode = beam_mode[subset]
    beam_mode.columns = list(uniquify(beam_mode.columns))
    if unique_subset:
        if subset != "all":
            beam_mode = beam_mode[subset]

    return beam_mode


def row_from_time(
    data: pd.DataFrame, t: TimeSpec, flatten: bool = False, **kwargs
) -> pd.Series:
    t = sanitize_t(t)
    if flatten:
        index = data.index.get_level_values("timestamp")
    else:
        index = data.index

    return data.iloc[index.get_loc(t, **kwargs)]


def coll_meta(augment_b2: bool = True) -> pd.DataFrame:
    coll_db_file = Path(__file__).parent / "metadata" / "coll_db.csv"
    df = pd.read_csv(coll_db_file, index_col="name")
    if augment_b2:
        df_b2 = df.copy()
        tmp = df_b2.index
        tmp = tmp.str.replace(r"\.B1", ".B2")
        tmp = tmp.str.replace(r"(?<=\d)R(?=\d)", "L~")
        tmp = tmp.str.replace(r"(?<=\d)L(?=\d)", "R~")
        tmp = tmp.str.replace("~", "")
        df_b2.index = tmp
        df = pd.concat([df, df_b2])
    return df


# TODO: this is broken for angles > pi
def angle_convert(angle):
    pi_2 = np.pi / 2
    if angle > pi_2:
        angle = pi_2 - (angle % pi_2)
    return angle / (pi_2)


def get_ADT(
    t1: TimeSpec,
    t2: TimeSpec,
    planes: List[str] = ["H", "V"],
    beams: List[int] = [1, 2],
    include: List[str] = ["amp", "length", "trigger"],
) -> pd.DataFrame:
    """Gets ADT blowup trigger data for the requested time interval, beam and plane.

    Args:
        t1: start of interval.
        t2: end of interval.
        planes: requested planes.
        beams: requested beams.
        include: list of ADT metrcis to fetch from timber. Must be a key of ADT_META.

    Returns:
        DataFrame as index the timestamp and columns the triggers of the beams/planes.
    """
    if not set(include) <= set(ADT_META.keys()):
        raise ValueError(f'"include" keys must be in {ADT_META.key()}')
    t1 = sanitize_t(t1)
    t2 = sanitize_t(t2)

    ADT_vars = []
    columns = []
    for plane in planes:
        for beam in beams:
            for v in sorted(include):
                ADT_vars.append(ADT_META[v].format(plane=plane, beam=beam))
                columns.append(f"ADT_B{beam}{plane}_{v}")

    data = DB.get(ADT_vars, t1, t2)
    dfs = []
    for var, d in data.items():
        df = pd.DataFrame(np.vstack(d).T, columns=["timestamp", var])
        df["timestamp"] = to_datetime(df["timestamp"].values)
        df = df.set_index("timestamp")
        dfs.append(df)

    # join them all on d[0]
    for i in dfs[1:]:
        dfs[0] = dfs[0].join(i, how="outer")

    joined = dfs[0]
    joined = joined.rename(mapper=dict(zip(ADT_vars, columns)), axis="columns")
    return joined


def sanitize_t(t: TimeSpec) -> pd.Timestamp:
    """Sanitizes input epoch or datetime string to pd.Timestamp in 'Europe/Zurich'
    timezone.

    Args:
        t:
            - int or float: assumes utc time, converts to pd.Timestamp and to
            Europe/Zurich timezone.

            - str: a pd.to_datetime compatible str, converts to pd.Timestamp
            and to 'Europe/Zurich' timezone if not already.

    Returns:
        Timestamp object for the given time.
    """

    if isinstance(t, (float, int)):
        t = to_datetime(t)
    elif isinstance(t, str):
        t = pd.to_datetime(t)
    if isinstance(t, pd.Timestamp):
        if t.tz is None:
            t = t.tz_localize("Europe/Zurich")
        elif t.tz.zone != "Europe/Zurich":
            t = t.tz_convert("Europe/Zurich")
    return t


def files_to_df(folder: PathSpec) -> pd.DataFrame:
    """Converts '%Y_%m_%d_%H_%M_%S%z' format file names into a dataframe with
    with the corresponding timestamp as index and filename as the only column.

    Args:
        folder: path to folder containing the time format compatible files.

    Returns:
        DataFrame: DatetimeIndexed DataFrame with a single column containing
            the file names.
    """
    folder = Path(folder)
    time_format = "%Y_%m_%d_%H_%M_%S%z"
    files = [file.name for file in folder.glob("*")]
    times = [
        pd.to_datetime(str(f).split(".")[0], format=time_format).tz_convert(
            "Europe/Zurich"
        )
        for f in files
    ]
    return (
        pd.DataFrame({"file": files, "timestamp": times})
        .set_index("timestamp")
        .sort_index()
    )


def no_limit_timber_get(variables: List[str], t1: TimeSpec, t2: TimeSpec, **kwargs):
    """Hacky bypass of the timber single query limit."""
    t1 = sanitize_t(t1)
    t2 = sanitize_t(t2)

    if not isinstance(variables, list):
        variables = [variables]

    try:
        out = DB.get(variables, t1, t2, **kwargs)
    except Exception:
        t12 = t1 + (t2 - t1) / 2
        out1 = no_limit_timber_get(variables, t1, t12, **kwargs)
        out2 = no_limit_timber_get(variables, t12, t2, **kwargs)
        out = {}
        for k in variables:
            out[k] = (
                np.hstack([out1[k][0], out2[k][0]]),
                np.vstack([out1[k][1], out2[k][1]]),
            )
    return out


# Helper class to handle wether to display the progress bar
class PBar:
    use_tqdm = True
    mute = False

    def __init__(self, iterable, **kwargs):

        self._tot = kwargs.get("total", None)
        if self._tot is None:
            try:
                self._tot = len(iterable)
            except AttributeError:
                self._tot = 0

        if self._tot > 1 and PBar.use_tqdm and not PBar.mute:
            iterable = tqdm(iterable, **kwargs)

        self._desc = kwargs.get("desc", "")
        self._msg = []
        self.iter = iterable

    def set_description(self, string):
        if not PBar.mute:
            end = ""
            if PBar.use_tqdm:
                if self._tot > 1:
                    self.iter.set_description(string)
                else:
                    end = "\n"
                    self._desc = string
            else:
                self._desc = string
            self._print_update(end=end)

    def __iter__(self):
        for n, i in enumerate(self.iter):
            if not PBar.use_tqdm and not PBar.mute:
                print("")
                n += 1
                msg = []
                if self._desc:
                    msg.append("{desc}")
                if self._tot > 1:
                    msg.append(f"{n} / {self._tot}")
                self._msg = msg

                if self._msg:
                    print(": ".join(self._msg).format(desc=self._desc), end="")
            yield i

    def _print_update(self, end=""):
        if "{desc}" not in self._msg:
            self._msg = ["{desc}"] + self._msg
        print("\r" + ": ".join(self._msg).format(desc=self._desc), end=end, flush=True)
