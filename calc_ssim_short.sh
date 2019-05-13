#!/bin/bash

ffmpeg -ss 0 -t 14 -i $1 -i $2 -lavfi  ssim -f null -
