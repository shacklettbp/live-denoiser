#!/bin/bash

rm outputs/*

if [ -z $1 ]; then
  weights=`ls -ct weights/*/*pth | head -1`
else
  weights="$1"
fi
echo "Using weights ${weights}"

python infer.py --weights=${weights} --inputs=../data/bistro/noisy1 --num-imgs=5999 --img-height=1080 --img-width=1920
