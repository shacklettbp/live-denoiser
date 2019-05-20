import matplotlib.pyplot as plt
import json
import os
import glob

num_frames = 750
dir = 'stats'

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
plt.xlabel("Frame #")
plt.ylabel("MS-SSIM")

for fname in glob.iglob(os.path.join(dir, "*.json")):
    with open(fname) as f:
        cur = json.load(f)
    name = os.path.splitext(os.path.basename(fname))[0]

    psnr = []
    ssim = []
    ms_ssim = []

    for i in range(num_frames):
        metrics = cur['frames'][i]['metrics']
        psnr.append(metrics['psnr'])
        ssim.append(metrics['ssim'])
        ms_ssim.append(metrics['ms_ssim'])

    plt.figure(0)
    plt.plot(range(num_frames), psnr, label=name)
    plt.figure(1)
    plt.plot(range(num_frames), ssim, label=name)
    plt.figure(2)
    plt.plot(range(num_frames), ms_ssim, label=name)

plt.figure(0)
box = psnr_ax.get_position()
psnr_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.figure(1)
box = ssim_ax.get_position()
ssim_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.figure(2)
box = ms_ssim_ax.get_position()
ms_ssim_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

psnr_graph.savefig("stats/psnr.png", bbox_inches='tight')
ssim_graph.savefig("stats/ssim.png", bbox_inches='tight')
ms_ssim_graph.savefig("stats/ms_ssim.png", bbox_inches='tight')
