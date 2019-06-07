#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import glob
import sys

matplotlib.rcParams['font.family'] = 'serif'

num_frames = 750

if len(sys.argv) < 2:
    print("Need path to stats")
    sys.exit(1)

dir = sys.argv[1]

psnr_graph = plt.figure(0)
psnr_ax = plt.subplot(111, label=0)
plt.xlabel("Frame #")
plt.ylabel("PSNR")

ssim_graph = plt.figure(1)
ssim_ax = plt.subplot(111, label=1)
plt.xlabel("Frame #")
plt.ylabel("SSIM")

ms_ssim_graph = plt.figure(2)
ms_ssim_ax = plt.subplot(111, label=2)
ms_ssim_ax.margins(0)
plt.xlabel("Frame #")
plt.ylabel("MS-SSIM")
plt.ylim(0.965, 0.995)

cmap = matplotlib.cm.get_cmap('Set1')

with open(os.path.join(dir, "desc")) as f:
    for line in f:
        title, style, color, fname = line.split(',')
        title, style, color, fname = title.strip(), style.strip(), int(color.strip()), fname.strip()

        print(title, style, color, fname)
        with open(os.path.join(dir, fname)) as cur:
            cur = json.load(cur)

        psnr = []
        ssim = []
        ms_ssim = []

        for i in range(num_frames):
            metrics = cur['frames'][i]['metrics']
            psnr.append(metrics['psnr'])
            ssim.append(metrics['ssim'])
            ms_ssim.append(metrics['ms_ssim'])

        plt.figure(0)
        plt.plot(range(num_frames), psnr, label=title, linewidth=0.8, color=cmap(color), linestyle=style)
        plt.figure(1)
        plt.plot(range(num_frames), ssim, label=title, linewidth=0.8, color=cmap(color), linestyle=style)
        plt.figure(2)
        plt.plot(range(num_frames), ms_ssim, label=title, linewidth=0.8, color=cmap(color), linestyle=style)

plt.figure(0)
box = psnr_ax.get_position()
psnr_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.figure(1)
box = ssim_ax.get_position()
ssim_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.figure(2)
box = ms_ssim_ax.get_position()
ms_ssim_ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.33))

psnr_graph.savefig(os.path.join(dir, "psnr.png"), bbox_inches='tight', dpi=250)
ssim_graph.savefig(os.path.join(dir, "ssim.png"), bbox_inches='tight', dpi=250)
ms_ssim_graph.savefig(os.path.join(dir, "ms_ssim.png"), bbox_inches='tight', dpi=250)
