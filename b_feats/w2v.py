import argparse
import psutil
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from utils import get_role
from paths import FEATS_DIR
from const import SEED

VECTOR_SIZE = 32  # size of embedding


def run_model(s):
    # construct sentences
    print('Constructing sentences')
    sentences, max_length = [], 0
    for idx in np.unique(s.index.values):
        sentence = s.loc[idx].values.tolist()
        sentences.append(sentence)
        max_length = np.maximum(max_length, len(sentence))
    # word2vec
    print('Training model')
    model = Word2Vec(sentences,
                     seed=SEED,
                     sg=1,
                     window=max_length,
                     min_count=1,
                     vector_size=VECTOR_SIZE,
                     workers=psutil.cpu_count())
    # output dataframe
    print('Creating output')
    leafs = model.wv.vocab.keys()
    output = pd.DataFrame(index=pd.Index([], name='leaf'),
                          columns=[str(i) for i in range(VECTOR_SIZE)])
    for leaf in leafs:
        output.loc[int(leaf)] = model.wv.get_vector(leaf)
    return output.sort_index()


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    byr = parser.parse_args().byr
    role = get_role(byr)

    # load sentences
    s = pd.read_pickle(FEATS_DIR + 'leaf_{}.pkl'.format(role))

    # run model
    df = run_model(s).rename(lambda x: role + x, axis=1).sort_index()

    # save
    df.to_pickle(FEATS_DIR + 'w2v_{}.pkl'.format(role))


if __name__ == '__main__':
    main()
