#!/bin/bash

if [ "$#" -lt 3 ]; then
  prefix="out"
else
  prefix="$3"
fi

ffmpeg -ss 0 -t 50 -apply_trc gamma28 -i $1/${prefix}_%d.exr -pix_fmt bgr24 -c:v libx264rgb -crf 0 $2
