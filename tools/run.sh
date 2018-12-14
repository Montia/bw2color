#!/usr/bin/env bash
WORK_DIR=/home/u22520/fromwin/bw2color

# 1. run spider
# 2. preprocess
# 1. generate tfrecord
# 1. train
if [ $1 == 1 ];then
    echo run spider $WORK_DIR/tools/spider.py
    python tools/spider.py --path $WORK_DIR/data/
elif [ $1 == 2 ];then
    echo preprocess
    python tools/preprocess.py --data $WORK_DIR/data/ --save $WORK_DIR/out/crop/
elif [ $1 == 3 ];then
    echo generate tfrecord
    python generateds.py --prefix $WORK_DIR
elif [ $1 == 4 ];then
    echo train
    python backward.py
fi

