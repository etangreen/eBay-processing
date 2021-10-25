import argparse
import multiprocessing as mp
import pickle
from time import sleep
import numpy as np
import pandas as pd
from constants import IDX, DAY, MAX_DELAY_TURN, NUM_CHUNKS
from paths import PARTS_DIR, PCTILE_DIR, FEATS_DIR
from constants import START, HOLIDAYS
from featnames import HOLIDAY, DOW_PREFIX, TIME_OF_DAY, AFTERNOON, LSTG, \
    CLOCK_FEATS, SLR, BYR


def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    with open(file, "rb") as f:
        obj = pickle.load(f)
    return obj


def topickle(contents=None, path=None):
    """
    Pickles a .pkl file encoded with Python 3
    :param contents: pickle-able object
    :param str path: path to file
    :return: contents of file
    """
    with open(path, "wb") as f:
        pickle.dump(contents, f, protocol=4)


def input_partition():
    """
    Parses command line input for partition name (and optional argument).
    :return part: string partition name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True, type=str)
    args = parser.parse_args()
    return args.part


def load_feats(name, lstgs=None, fill_zero=False):
    """
    Loads dataframe of features (and reindexes).
    :param str name: filename
    :param lstgs: listings to restrict to
    :param bool fill_zero: fill missings with 0's if True
    :return: dataframe of features
    """
    df = unpickle(FEATS_DIR + '{}.pkl'.format(name))
    if lstgs is None:
        return df
    kwargs = {'index': lstgs}
    if len(df.index.names) > 1:
        kwargs['level'] = LSTG
    if fill_zero:
        kwargs['fill_value'] = 0.
    return df.reindex(**kwargs)


def extract_clock_feats(seconds):
    """
    Creates clock features from timestamps.
    :param seconds: seconds since START.
    :return: tuple of time_of_day sine transform and afternoon indicator.
    """
    sec_norm = (seconds % DAY) / DAY
    time_of_day = np.sin(sec_norm * np.pi)
    afternoon = sec_norm >= 0.5
    return time_of_day, afternoon


def extract_day_feats(seconds):
    """
    Creates clock features from timestamps.
    :param seconds: seconds since START.
    :return: tuple of time_of_day sine transform and afternoon indicator.
    """
    clock = pd.to_datetime(seconds, unit='s', origin=START)
    df = pd.DataFrame(index=clock.index)
    df[HOLIDAY] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df[DOW_PREFIX + str(i)] = clock.dt.dayofweek == i
    return df


def collect_date_clock_feats(seconds):
    """
    Combines date and clock features.
    :param seconds: seconds since START.
    :return: dataframe of date and clock features.
    """
    df = extract_day_feats(seconds)
    df[TIME_OF_DAY], df[AFTERNOON] = extract_clock_feats(seconds)
    assert list(df.columns) == CLOCK_FEATS
    return df


def get_days_delay(clock):
    """
    Calculates time between successive offers.
    :param clock: dataframe with index ['lstg', 'thread'],
        turn numbers as columns, and seconds since START as values
    :return days: fractional number of days between offers.
    :return delay: time between offers as share of MAX_DELAY.
    """
    # initialize output dataframes in wide format
    days = pd.DataFrame(0., index=clock.index, columns=clock.columns)
    delay = pd.DataFrame(0., index=clock.index, columns=clock.columns)

    # for turn 1, days and delay are 0
    for i in range(2, 8):
        days[i] = clock[i] - clock[i - 1]
        delay[i] = days[i] / MAX_DELAY_TURN
    # no delay larger than 1
    assert delay.max().max() <= 1

    # reshape from wide to long
    days = days.rename_axis('index', axis=1).stack() / DAY
    delay = delay.rename_axis('index', axis=1).stack()

    return days, delay


def slr_norm(con=None, prev_byr_norm=None, prev_slr_norm=None):
    """
    Normalized offer for seller turn.
    :param con: current concession, between 0 and 1.
    :param prev_byr_norm: normalized concession from one turn ago.
    :param prev_slr_norm: normalized concession from two turns ago.
    :return: normalized distance of current offer from start_price to 0.
    """
    return 1 - con * prev_byr_norm - (1 - prev_slr_norm) * (1 - con)


def byr_norm(con=None, prev_byr_norm=None, prev_slr_norm=None):
    """
    Normalized offer for buyer turn.
    :param con: current concession, between 0 and 1.
    :param prev_byr_norm: normalized concession from two turns ago.
    :param prev_slr_norm: normalized concession from one turn ago.
    :return: normalized distance of current offer from 0 to start_price.
    """
    return (1 - prev_slr_norm) * con + prev_byr_norm * (1 - con)


def get_norm(con):
    """
    Calculate normalized concession from rounded concessions.
    :param con: pandas series of rounded concessions.
    :return: pandas series of normalized concessions.
    """
    df = con.unstack()
    norm = pd.DataFrame(index=df.index, columns=df.columns)
    norm[1] = df[1]
    norm[2] = df[2] * (1 - norm[1])
    for i in range(3, 8):
        if i in IDX[BYR]:
            norm[i] = byr_norm(con=df[i],
                               prev_byr_norm=norm[i - 2],
                               prev_slr_norm=norm[i - 1])
        elif i in IDX[SLR]:
            norm[i] = slr_norm(con=df[i],
                               prev_byr_norm=norm[i - 1],
                               prev_slr_norm=norm[i - 2])
    return norm.rename_axis('index', axis=1).stack().astype('float64')


def get_lstgs(part):
    """
    Grabs list of partition-specific listing ids.
    :param str part: name of partition
    :return: list of partition-specific listings ids
    """
    d = unpickle(PARTS_DIR + 'partitions.pkl')
    return d[part]


def do_rounding(price):
    """
    Returns booleans for whether offer is round and ends in nines
    :param price: price in dollars and cents
    :return: (bool, bool)
    """
    assert np.min(price) >= .01
    cents = np.round(100 * (price % 1)).astype(np.int8)
    dollars = np.floor(price).astype(np.int64)
    is_nines = (cents >= 90) | np.isin(dollars, [9, 99] + list(range(990, 1000)))
    is_round = (cents == 0) & ~is_nines & \
               (((dollars <= 100) & (dollars % 5 == 0)) | (dollars % 50 == 0))
    assert np.all(~(is_round & is_nines))
    return is_round, is_nines


def load_pctile(name=None):
    """
    Loads the percentile file given by name.
    :param str name: name of feature
    :return: pd.Series with feature values in the index and percentiles in the values
    """
    path = PCTILE_DIR + '{}.pkl'.format(name)
    return unpickle(path)


def feat_to_pctile(s=None, pc=None):
    """
    Converts byr hist counts to percentiles or visa versa.
    :param pandas.Series s: counts
    :param pandas.Series pc: percentiles
    :return: Series
    """
    if pc is None:
        pc = load_pctile(name=str(s.name))
    v = pc.reindex(index=s.values, method='pad').values
    return pd.Series(v, index=s.index, name=s.name)


def run_func_on_chunks(f=None, func_kwargs=None, num_chunks=NUM_CHUNKS):
    """
    Applies f to all chunks in parallel.
    :param f: function that takes chunk number as input along with
    other arguments
    :param func_kwargs: dictionary of other keyword arguments
    :param int num_chunks: number of chunks
    :return: list of worker-specific output
    """
    num_workers = min(num_chunks, mp.cpu_count())
    print('Using {} workers'.format(num_workers))
    pool = mp.Pool(num_workers)
    jobs = []
    for i in range(num_chunks):
        kw = func_kwargs.copy()
        kw['chunk'] = i
        jobs.append(pool.apply_async(f, kwds=kw))
    res = []
    for job in jobs:
        while True:
            if job.ready():
                res.append(job.get())
                break
            else:
                sleep(5)
    return res


def process_chunk_worker(part=None, chunk=None, gen_class=None, gen_kwargs=None):
    if gen_kwargs is None:
        gen_kwargs = dict()
    gen = gen_class(**gen_kwargs)
    return gen.process_chunk(chunk=chunk, part=part)
