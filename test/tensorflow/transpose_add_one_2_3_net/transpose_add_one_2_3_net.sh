#!/bin/sh
set -e
wget https://github.com/tensorflow/tensorflow/raw/master/tensorflow/python/tools/freeze_graph.py \
     -O freeze_graph.py
./transpose_add_one_net.py -r 2 -c 3
