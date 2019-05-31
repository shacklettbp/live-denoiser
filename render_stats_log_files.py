#!/usr/bin/env python

import json
import os
import glob
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

fstart = 0
fstop  = 894
dir = 'stats'

# psnr_graph = plt.figure(0)
# psnr_ax = plt.subplot(111, label=0)
# plt.xlabel("Frame #")
# plt.ylabel("PSNR")

ssim_graph = plt.figure(0)
ssim_ax = plt.subplot(111, label=1)
plt.xlabel("Frame #")
plt.ylabel("SSIM")

# ms_ssim_graph = plt.figure(2)
# ms_ssim_ax = plt.subplot(111, label=2)
# plt.xlabel("Frame #")
# plt.ylabel("MS-SSIM")

for fname in sorted(glob.iglob(os.path.join(dir, "*_ssim.log"))):
    with open(fname) as f:

        frames = []
        ssims = []

        for line in f:
            items = line.split(" ")
            fid = 0
            ssim = 0.0
            for item in items:
                keyval = item.split(":")

                if keyval[0] == "n":
                    fid = int(keyval[1])
                if keyval[0] == "All":
                    ssim = float(keyval[1])

            if fid >= fstart and fid < fstop:
                frames.append(fid)
                ssims.append(ssim)

        plt.figure(0)
        # plt.plot(range(num_frames), psnr, label=name)
        # plt.figure(1)
        plt.plot(frames, ssims, linewidth=1.0, label=fname)
        # plt.figure(2)
        # plt.plot(range(num_frames), ms_ssim, label=name)

plt.figure(0)
# box = psnr_ax.get_position()
# psnr_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# plt.figure(1)
box = ssim_ax.get_position()
ssim_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# plt.figure(2)
# box = ms_ssim_ax.get_position()
# ms_ssim_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# psnr_graph.savefig("stats/psnr.png", bbox_inches='tight')
ssim_graph.savefig("stats/ssim.pdf", bbox_inches='tight')
ssim_graph.savefig("stats/ssim.png", bbox_inches='tight')
# ms_ssim_graph.savefig("stats/ms_ssim.png", bbox_inches='tight')
