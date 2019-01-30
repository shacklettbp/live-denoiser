import sys
import os
import json
import torch
import argparse

from dataset import get_files
from data_loading import load_exr, dump_raw
from utils import tonemap
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument('--noisy', type=str, required=True)
parser.add_argument('--ref', type=str, default=None)
parser.add_argument('--dst', type=str, required=True)
parser.add_argument('--num-imgs', type=int, required=True)
parser.add_argument('--num-temporal', type=int, default=2)
parser.add_argument('--divide-data', action='store_true', default=False)
parser.add_argument('--single-reference', action='store_true', default=False)
parser.add_argument('--no-crop', action='store_true', default=False)
args = parser.parse_args()

total_num_imgs = int(args.num_imgs)

num_imgs = total_num_imgs
if args.divide_data:
    num_imgs //= 2

data_dir = args.noisy
dst_dir = args.dst
files = get_files(data_dir, 'exr', total_num_imgs)

if args.single_reference:
    assert(args.num_temporal == 0)
    ref_file = args.ref
else:
    ref_files = get_files(args.ref, 'exr', num_imgs) if args.ref is not None else None

_, height, width = load_exr(files[0][0]).shape

if args.no_crop:
    cropsize = (width, height)
else:
    cropsize = (256, 256)

def get_crops(hdr, normal, albedo, ref):
    hdr[torch.isnan(hdr)] = 0 # NaNs
    ref[torch.isnan(ref)] = 0
    albedo[torch.isnan(albedo)] = 0
    normal[torch.isnan(normal)] = 0

    crops = []
    for y in chain(range(0, height, cropsize[1])[:-1], [height - cropsize[1]]):
        for x in chain(range(0, width, cropsize[0])[:-1], [width - cropsize[0]]):
            hdr_crop = hdr[0][:, y:y+cropsize[1], x:x+cropsize[0]]
            normal_crop = normal[0][:, y:y+cropsize[1], x:x+cropsize[0]]
            albedo_crop = albedo[0][:, y:y+cropsize[1], x:x+cropsize[0]]
            ref_crop = ref[0][:, y:y+cropsize[1], x:x+cropsize[0]]

            if torch.abs(hdr_crop - ref_crop).sum() == 0: # Skyboxes
                continue

            hdr_crop = [hdr_crop]
            normal_crop = [normal_crop]
            albedo_crop = [albedo_crop]
            ref_crop = [ref_crop]
            for i in range(1, args.num_temporal+1):
                hdr_crop.append(hdr[i][:, y:y+cropsize[1], x:x+cropsize[0]])
                normal_crop.append(normal[i][:, y:y+cropsize[1], x:x+cropsize[0]])
                albedo_crop.append(albedo[i][:, y:y+cropsize[1], x:x+cropsize[0]])
                ref_crop.append(ref[i][:, y:y+cropsize[1], x:x+cropsize[0]])

            crops.append((hdr_crop, normal_crop, albedo_crop, ref_crop))

    return crops

filenames = []

for idx in range(0, num_imgs - args.num_temporal, args.num_temporal + 1):
    hdr = []
    normal = []
    albedo = []
    ref = []
    for i in range(args.num_temporal + 1):
        hdr_fn, normal_fn, albedo_fn, ref_fn = files[idx + i]
        if args.single_reference:
            ref_fn = ref_file
        elif ref_files is not None:
            ref_fn, _, _, _  = ref_files[idx + i]
        if args.divide_data:
            ref_fn, _, _, _ = files[num_imgs + idx + i]

        hdr.append(load_exr(hdr_fn))
        normal.append(load_exr(normal_fn))
        albedo.append(load_exr(albedo_fn))
        ref.append(load_exr(ref_fn))

    crops = get_crops(torch.stack(hdr), torch.stack(normal), torch.stack(albedo), torch.stack(ref))

    crop_filenames = []
    for crop_idx, (hdr_crop, normal_crop, albedo_crop, ref_crop) in enumerate(crops):
        temporal_filenames = []
        for temporal_idx in range(args.num_temporal + 1):
            hdr_name = 'hdr_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            normal_name = 'normal_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            albedo_name = 'albedo_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            ref_name = 'ref_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            dump_raw(hdr_crop[temporal_idx].cpu(), os.path.join(dst_dir, hdr_name))
            dump_raw(normal_crop[temporal_idx].cpu(), os.path.join(dst_dir, normal_name))
            dump_raw(albedo_crop[temporal_idx].cpu(), os.path.join(dst_dir, albedo_name))
            dump_raw(ref_crop[temporal_idx].cpu(), os.path.join(dst_dir, ref_name))

            temporal_filenames.append((hdr_name, normal_name, albedo_name, ref_name))

        crop_filenames.append(temporal_filenames)
    filenames.append(crop_filenames)

metadata = {}
metadata['files'] = filenames
metadata['size'] = cropsize

with open(os.path.join(dst_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f)
