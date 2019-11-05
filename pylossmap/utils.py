import pytimber
import numpy as np
import pandas as pd
from pathlib import Path

from . import timber_vars

DB = pytimber.LoggingDB()

# TODO: Figure out how to get the accelerator mode.
#                              Timber var,               timeseries
BEAM_META = {'intensity':      (timber_vars.INTENSITY,   True),
             'filling_scheme': (timber_vars.FILL_SCHEME, False),
             'number_bunches': (timber_vars.BUNCH_NUM,   False),
             'energy':         (timber_vars.ENERGY,      True),
             # 'amode':          (timber_vars.ACCEL_MODE,  False),
             # 'bmode':          (timber_vars.BEAM_MODE,   False),
             }


def uniquify(iterable):
    '''Makes the entries in a list unique.

    Args:
        iterable (Iterable): list to uniquify, duplicates will have
        "_{number}" added to them.

    Yields:
        str: uniquified element of iterable
    '''
    seen = set()
    for item in iterable:
        fudge = 1
        newitem = item
        while newitem in seen:
            fudge += 1
            newitem = "{}_{}".format(item, fudge)
        yield newitem
        seen.add(newitem)


def to_datetime(ts):
    return pd.to_datetime(ts,
                          unit='s',
                          utc=True).tz_convert('Europe/Zurich')


def fill_from_time(t, fuzzy_t='12H'):
    '''Gets the machine fill of a timestamp.

    Returns:
        dict: dict containing the start/end time of the fill and
        with beam mode info.
    '''
    fuzzy_t = pd.Timedelta(fuzzy_t)

    fills = []
    i = 1
    while fills == []:
        fills = DB.getLHCFillsByTime(t - i*fuzzy_t,
                                     t + i*fuzzy_t)
        i += 1

    for fill in fills:
        if to_datetime(fill['startTime']) <= t and t <= to_datetime(fill['endTime']):
            return fill
    raise ValueError('Fill not found.')


def beammode_from_time(t, fill=None, **kwargs):
    if fill is None:
        fill = fill_from_time(t, **kwargs)

    for bm in fill['beamModes']:
        if t >= to_datetime(bm['startTime']) and t <= to_datetime(bm['endTime']):
            return bm
    return fill


def beammode_to_df(beam_mode, subset='all', unique_subset=False):
    # put beam mode timestamps into dataframe
    beam_mode = pd.DataFrame(beam_mode)
    beam_mode = beam_mode.set_index('mode').T
    beam_mode = beam_mode.applymap(to_datetime)

    if not unique_subset:
        if subset != 'all':
            beam_mode = beam_mode[subset]
    beam_mode.columns = list(uniquify(beam_mode.columns))
    if unique_subset:
        if subset != 'all':
            beam_mode = beam_mode[subset]

    return beam_mode


def row_from_time(data, t, flatten=False, **kwargs):
    if flatten:
        index = data.index.get_level_values('timestamp')
    else:
        index = data.index

    return data.iloc[index.get_loc(t, **kwargs)]


def coll_meta(augment_b2=True):
    coll_db_file = Path(__file__).parent / 'metadata' / 'coll_db.csv'
    df = pd.read_csv(coll_db_file, index_col='name')
    if augment_b2:
        df_b2 = df.copy()
        tmp = df_b2.index
        tmp = tmp.str.replace(r'\.B1', '.B2')
        tmp = tmp.str.replace(r'(?<=\d)R(?=\d)', 'L~')
        tmp = tmp.str.replace(r'(?<=\d)L(?=\d)', 'R~')
        tmp = tmp.str.replace('~', '')
        df_b2.index = tmp
        df = pd.concat([df, df_b2])
    return df


def angle_convert(angle):
    pi_2 = np.pi/2
    if angle > pi_2:
        angle = pi_2 - (angle % pi_2)
    return angle/(pi_2)


def get_ADT(t1, t2, planes=['H', 'V'], beams=[1, 2]):
    """Gets ADT blowup trigger data for the requested time interval,
    beam and plane.

    Args:
        t1 (Datetime): start of interval.
        t2 (Datetime): end of interval.
        planes (list, optional): requested planes.
        beams (list, optional): requested beams,

    Returns:
        DataFrame: DataFrame as index the timestamp and columns the
        triggers of the beams/planes.
    """
    ADT_vars = []
    columns = []
    for plane in planes:
        for beam in beams:
            ADT_vars.append(timber_vars.ADT_TRIGGER.format(plane=plane,
                                                           beam=beam))
            columns.append(f'ADT_B{beam}{plane}')

    data = DB.get(ADT_vars,
                  t1,
                  t2)
    dfs = []
    for (var, d), c in zip(data.items(), columns):
        df = pd.DataFrame(np.vstack(d).T, columns=['timestamps', c])
        df['timestamps'] = to_datetime(df['timestamps'].values)
        df = df.set_index('timestamps')
        dfs.append(df)

    # join them all on d[0]
    for i in dfs[1:]:
        dfs[0] = dfs[0].join(i, how='outer')

    joined = dfs[0]
    return joined


def sanitize_t(t):
    if isinstance(t, (float, int)):
        t = to_datetime(t)
    elif isinstance(t, pd.Timestamp):
        if t.tz is None:
            t = t.tz_localize('Europe/Zurich')
        elif t.tz.zone != 'Europe/Zurich':
            t = t.tz_convert('Europe/Zurich')
    return t
