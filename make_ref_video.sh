#!/bin/bash

ffmpeg -ss 0 -t 50 -apply_trc gamma28 -i $1/out_%d.exr -pix_fmt yuv420p -c:v libx264 -crf 0 $2
