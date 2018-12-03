#!/bin/bash

rm outputs/*

if [ -z $1 ]; then
  weights=`ls -ct weights/*/*pth | head -1`
else
  weights="$1"
fi
echo "Using weights ${weights}"

python infer.py --weights=${weights} --inputs=../data/suntemple/noisy_good1 --num-imgs=1000 --img-height=1080 --img-width=1920
