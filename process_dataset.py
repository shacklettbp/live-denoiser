from dataset import get_files
from data_loading import load_exr, dump_raw
from utils import tonemap
import sys
import os
import torch
from itertools import chain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--noisy', type=str, required=True)
parser.add_argument('--ref', type=str, default=None)
parser.add_argument('--dst', type=str, required=True)
parser.add_argument('--num-imgs', type=int, required=True)
parser.add_argument('--num-temporal', type=int, default=2)
parser.add_argument('--divide-data', action='store_true', default=False)
args = parser.parse_args()

total_num_imgs = int(args.num_imgs)

num_imgs = total_num_imgs
if args.divide_data:
    num_imgs //= 2

data_dir = args.noisy
dst_dir = args.dst
files = get_files(data_dir, 'exr', total_num_imgs)
ref_files = get_files(args.ref, 'exr', num_imgs) if args.ref is not None else None
cropsize = 256

def get_crops(hdr, normal, albedo, ref):
    hdr[torch.isnan(hdr)] = 0 # NaNs
    ref[torch.isnan(ref)] = 0
    albedo[torch.isnan(albedo)] = 0
    normal[torch.isnan(normal)] = 0


    _, height, width = hdr[0].shape
    crops = []
    for y in chain(range(0, height, cropsize)[:-1], [height - cropsize]):
        for x in chain(range(0, width, cropsize)[:-1], [width - cropsize]):
            hdr_crop = hdr[0][:, y:y+cropsize, x:x+cropsize]
            normal_crop = normal[0][:, y:y+cropsize, x:x+cropsize]
            albedo_crop = albedo[0][:, y:y+cropsize, x:x+cropsize]
            ref_crop = ref[0][:, y:y+cropsize, x:x+cropsize]

            if torch.abs(hdr_crop - ref_crop).sum() == 0: # Skyboxes
                continue

            hdr_crop = [hdr_crop]
            normal_crop = [normal_crop]
            albedo_crop = [albedo_crop]
            ref_crop = [ref_crop]
            for i in range(1, args.num_temporal+1):
                hdr_crop.append(hdr[i][:, y:y+cropsize, x:x+cropsize])
                normal_crop.append(normal[i][:, y:y+cropsize, x:x+cropsize])
                albedo_crop.append(albedo[i][:, y:y+cropsize, x:x+cropsize])
                ref_crop.append(ref[i][:, y:y+cropsize, x:x+cropsize])

            crops.append((hdr_crop, normal_crop, albedo_crop, ref_crop))

    return crops

for idx in range(0, num_imgs - args.num_temporal, args.num_temporal + 1):
    hdr = []
    normal = []
    albedo = []
    ref = []
    for i in range(args.num_temporal + 1):
        hdr_fn, normal_fn, albedo_fn, ref_fn = files[idx + i]
        if ref_files is not None:
            ref_fn, _, _, _  = ref_files[idx + i]
        if args.divide_data:
            ref_fn, _, _, _ = files[num_imgs + idx + i]

        hdr.append(load_exr(hdr_fn))
        normal.append(load_exr(normal_fn))
        albedo.append(load_exr(albedo_fn))
        ref.append(load_exr(ref_fn))

    crops = get_crops(torch.stack(hdr), torch.stack(normal), torch.stack(albedo), torch.stack(ref))

    for crop_idx, (hdr_crop, normal_crop, albedo_crop, ref_crop) in enumerate(crops):
        for temporal_idx in range(args.num_temporal + 1):
            dump_raw(hdr_crop[temporal_idx].cpu(), os.path.join(dst_dir, 'hdr_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)))
            dump_raw(normal_crop[temporal_idx].cpu(), os.path.join(dst_dir, 'normal_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)))
            dump_raw(albedo_crop[temporal_idx].cpu(), os.path.join(dst_dir, 'albedo_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)))
            dump_raw(ref_crop[temporal_idx].cpu(), os.path.join(dst_dir, 'ref_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)))
