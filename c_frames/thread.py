import pandas as pd
from utils import get_lstgs, get_days_since_lstg, input_partition, load_feats
from constants import MAX_DAYS, DAY
from paths import PARTS_DIR
from featnames import BYR_HIST, DAYS_SINCE_LSTG, START_DATE, INDEX, X_THREAD


def create_x_thread(lstgs=None):
    # load data
    offers = load_feats('offers', lstgs=lstgs)
    thread_start = offers.clock.xs(1, level=INDEX)
    start_date = load_feats('listings', lstgs=lstgs)[START_DATE]
    lstg_start = start_date.astype('int64') * DAY
    hist = load_feats('threads', lstgs=lstgs)[BYR_HIST]

    # days since lstg start
    days = get_days_since_lstg(lstg_start, thread_start)
    days = days.rename(DAYS_SINCE_LSTG)
    assert days.max() < MAX_DAYS

    # create dataframe
    x_thread = pd.concat([days, hist], axis=1)

    return x_thread


def main():
    part = input_partition()
    print('{}/x_thread'.format(part))

    x_thread = create_x_thread(lstgs=get_lstgs(part))
    x_thread.to_pickle(PARTS_DIR + '{}/{}.pkl'.format(part, X_THREAD))


if __name__ == "__main__":
    main()
