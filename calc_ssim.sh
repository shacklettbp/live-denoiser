#!/bin/bash

ffmpeg -i $1 -i $2 -lavfi  ssim -f null -
