import pandas as pd
from util import collect_date_clock_feats, get_days_delay, get_lstgs, get_common_cons
from utils import input_partition, load_feats
from constants import IDX
from paths import PARTS_DIR
from featnames import DAYS, DELAY, CON, NORM, COMMON, MSG, REJECT, \
    AUTO, EXP, SLR, CLOCK, INDEX


def get_x_offer(offers, tf):
    # initialize output dataframe
    df = pd.DataFrame(index=offers.index).sort_index()

    # clock features
    df = df.join(collect_date_clock_feats(offers[CLOCK]))

    # differenced time feats
    df = df.join(tf.reindex(index=df.index, fill_value=0))

    # delay features
    df[DAYS], df[DELAY] = get_days_delay(offers[CLOCK].unstack())

    # auto and exp are functions of delay
    df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR], level=INDEX)
    df[EXP] = df[DELAY] == 1

    # concession
    df.loc[:, [CON, REJECT, NORM]] = offers.loc[:, [CON, REJECT, NORM]]

    # common concessions
    df[COMMON] = get_common_cons(con=df[CON])

    # message indicator is last
    df[MSG] = offers.message

    # error checking
    assert all(df.loc[df[EXP], REJECT])
    assert all(~df.loc[df[CON] == 0, COMMON])
    assert all(~df.loc[df[CON] == 1, COMMON])

    return df


def create_x_offer(lstgs=None):
    # load data
    offers = load_feats('offers', lstgs=lstgs)
    tf = load_feats('tf', lstgs=lstgs)

    # offer features
    x_offer = get_x_offer(offers, tf)

    return x_offer, offers.clock


def main():
    part = input_partition()
    print('{}/x_offer'.format(part))

    x_offer, clock = create_x_offer(lstgs=get_lstgs(part))
    x_offer.to_pickle(PARTS_DIR + '{}/x_offer.pkl'.format(part))
    clock.to_pickle(PARTS_DIR + '{}/clock.pkl'.format(part))


if __name__ == "__main__":
    main()
