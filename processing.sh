#!/bin/bash

# create features
python repo/a_sundry.py
python repo/b_feats/other.py
python repo/b_feats/w2v.py
python repo/b_feats/w2v.py --slr
python repo/b_feats/category.py --name leaf &
python repo/b_feats/category.py --name meta
python repo/b_feats/category.py --name slr
python repo/b_feats/tf.py

# create frames
for part in sim rl valid
do
  for f in lookup lstg thread offer
  do
    python repo/c_frames/$f\.py --part $part &
  done
done