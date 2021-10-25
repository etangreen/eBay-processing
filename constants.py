from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from featnames import SLR, BYR, TRAIN_MODELS, TRAIN_RL, VALIDATION

# for splitting data
SHARES = {TRAIN_MODELS: 0.75, TRAIN_RL: 0.1, VALIDATION: 0.05}

# listing window stays open this many days
MAX_DAYS = 8

# temporal constants
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR

# indices for byr and slr offers
IDX = {
    BYR: [1, 3, 5, 7],
    SLR: [2, 4, 6]
}

# date range and holidays
START = '2012-06-01 00:00:00'
END = '2013-05-31 23:59:59'

# maximal delay times
MAX_DELAY_TURN = 2 * DAY
HOLIDAYS = Calendar().holidays(start=START, end=END)

# fixed random seed
SEED = 123456

# number of chunks
NUM_CHUNKS = 1024

# number of concessions available to agent
NUM_COMMON_CONS = 6
