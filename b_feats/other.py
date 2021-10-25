import pandas as pd
from utils import topickle
from constants import MAX_DAYS, START, END, HOLIDAYS
from paths import FEATS_DIR
from featnames import HOLIDAY, DOW_PREFIX


def main():
    num = pd.to_datetime(END) - pd.to_datetime(START) \
          + pd.to_timedelta(MAX_DAYS, unit='d') \
          + pd.to_timedelta(1, unit='s')
    days = pd.to_datetime(list(range(num.days)), unit='D', origin=START)
    df = pd.DataFrame(index=days)
    df[HOLIDAY] = days.isin(HOLIDAYS)
    for i in range(6):
        df[DOW_PREFIX + str(i)] = days.dayofweek == i
    topickle(df.values, FEATS_DIR + 'date_feats.pkl')


if __name__ == "__main__":
    main()
