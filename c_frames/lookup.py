import pandas as pd
from utils import get_lstgs, input_partition, load_feats
from constants import DAY
from paths import PARTS_DIR
from featnames import LOOKUP, START_TIME, END_TIME, START_PRICE, DEC_PRICE, \
    ACC_PRICE, START_DATE


def create_lookup(lstgs=None):
    # load data
    listings = load_feats('listings', lstgs=lstgs)

    # start time instead of start date
    start_time = listings[START_DATE].astype('int64') * DAY
    start_time = start_time.rename(START_TIME)

    # subset features
    lookup = listings[[START_PRICE, DEC_PRICE, ACC_PRICE]]
    lookup = pd.concat([lookup, start_time, listings[END_TIME]], axis=1)

    return lookup


def main():
    part = input_partition()
    print('{}/{}'.format(part, LOOKUP))

    lookup = create_lookup(lstgs=get_lstgs(part))
    lookup.to_pickle(PARTS_DIR + '{}/{}.pkl'.format(part, LOOKUP))


if __name__ == "__main__":
    main()
