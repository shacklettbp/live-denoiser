#!/bin/bash

ffmpeg -ss 0 -t 30 -i $1 -ss 0 -t 30 -i $2 -lavfi "ssim=stats/$3_ssim.log;[0:v][1:v]psnr=stats/$3_psnr.log" -f null -
