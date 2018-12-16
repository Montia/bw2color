#!/usr/bin/env bash

server="colfax"
dir="/home/u22520/fromwin/bw2color/"

# 1. upload all project file
# 2. download latest 10 training_result image
# 3. download model
if [ $1 == 1 ];then
    scp backward.py forward.py generateds.py test.py $server:$dir
    scp -r tools/ $server:$dir
elif [ $1 == 2 ];then
    path=${dir}training_result/
    echo download latest 10 images from $path
    images=$(ssh $server 'ls -t '$path' | head -n 10')
    for img in $images
    do
        scp $server:$path$img ./training_result
    done
fi