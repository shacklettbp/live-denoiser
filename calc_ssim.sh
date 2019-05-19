#!/bin/bash

ffmpeg -ss 0 -t 30 -i $1 -ss 0 -t 26 -i $2 -lavfi "ssim=ssim.log;[0:v][1:v]psnr=psnr.log" -f null -
