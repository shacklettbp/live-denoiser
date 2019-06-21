#!/bin/bash

if [ "$#" -lt 3 ]; then
  prefix="out"
else
  prefix="$3"
fi

ffmpeg -ss 0 -t 50 -apply_trc gamma28 -i $1/out_%d.exr -pix_fmt yuv420p -c:v libx264 -crf 5 $2

