import os


DATA_DIR = '/data/eBay/'                  # where data are stored
CLEAN_DIR = DATA_DIR + 'clean/'           # houses csvs
PARTS_DIR = DATA_DIR + 'partitions/'      # post-partition features
PCTILE_DIR = DATA_DIR + 'pctile/'         # percentiles of features
FEATS_DIR = DATA_DIR + 'feats/'           # pre-partion features
FEATNAMES_DIR = DATA_DIR + 'inputs/featnames/'  # for testing

folders = [elem for elem in globals().values()
           if type(elem) is str and elem.startswith(DATA_DIR)]
for folder in folders:
    if type(folder) is str and not os.path.isdir(folder):
        os.makedirs(folder)
