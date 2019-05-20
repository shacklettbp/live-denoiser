#!/bin/bash

~/rendering/vmaf-build/bin/ffmpeg -ss 0 -t 30 -i $1 -ss 0 -t 30 -i $2 -lavfi libvmaf="log_fmt=json:log_path=stats/$3.json:psnr=1:ssim=1:ms_ssim=1" -f null -
