# strings for referencing quantities related to buyer and seller interface
SLR = 'slr'
BYR = 'byr'

# clock feats
HOLIDAY = 'holiday'
DOW_PREFIX = 'dow'
TIME_OF_DAY = 'time_of_day'
AFTERNOON = 'afternoon'

DATE_FEATS = [HOLIDAY] + [DOW_PREFIX + str(i) for i in range(6)]
CLOCK_FEATS = DATE_FEATS + [TIME_OF_DAY, AFTERNOON]

# outcomes
DAYS = 'days'
DELAY = 'delay'
CON = 'con'
NORM = 'norm'
COMMON = 'common'
MSG = 'msg'
ACCEPT = 'accept'
REJECT = 'reject'
AUTO = 'auto'
EXP = 'exp'

# thread features
BYR_HIST = 'byr_hist'
DAYS_SINCE_LSTG = 'days_since_lstg'

# index labels
LSTG = 'lstg'
THREAD = 'thread'
INDEX = 'index'
CLOCK = 'clock'

# lookup column names
META = 'meta'
LEAF = 'leaf'
CNDTN = 'cndtn'
START_DATE = 'start_date'
START_TIME = 'start_time'
END_TIME = 'end_time'
START_PRICE = 'start_price'
DEC_PRICE = 'decline_price'
ACC_PRICE = 'accept_price'
SLR_BO_CT = 'slr_bo_ct'
STORE = 'store'

# partition components
X_LSTG = 'x_lstg'
X_OFFER = 'x_offer'
X_THREAD = 'x_thread'
LOOKUP = 'lookup'

# partitions
TRAIN_MODELS = 'sim'
TRAIN_RL = 'rl'
VALIDATION = 'valid'
TEST = 'test'
