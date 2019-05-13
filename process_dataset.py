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
parser.add_argument('--ref', type=str, required=True)
parser.add_argument('--dst', type=str, required=True)
parser.add_argument('--num-imgs', type=int, required=True)
parser.add_argument('--num-temporal', type=int, default=2)
parser.add_argument('--no-crop', action='store_true', default=False)
parser.add_argument('--extra', action='store_true', default=False)
args = parser.parse_args()

total_num_imgs = int(args.num_imgs)

num_imgs = total_num_imgs

data_dir = args.noisy
dst_dir = args.dst
files = get_files(data_dir, 'exr', total_num_imgs)
ref_files = get_files(args.ref, 'exr', num_imgs)

_, height, width = load_exr(files[0][0]).shape

if args.no_crop:
    cropsize = (width, height)
else:
    cropsize = (256, 256)

def get_crops(features):
    features[torch.isnan(features)] = 0 # Nans

    crops = []
    for y in chain(range(0, height, cropsize[1])[:-1], [height - cropsize[1]]):
        for x in chain(range(0, width, cropsize[0])[:-1], [width - cropsize[0]]):
            crop = features[:, :, y:y+cropsize[1], x:x+cropsize[0]]

            if torch.abs(crop[0, 0:3, ...] - crop[0:, 8:11, ...]).sum() == 0: # Skyboxes
                continue

            crops.append(crop)

    return crops

filenames = []

for idx in range(0, num_imgs - args.num_temporal, args.num_temporal + 1):
    temporal = []

    for i in range(args.num_temporal + 1):
        hdr_fn, normal_fn, albedo_fn, direct_fn, indirect_fn, shadowt_fn = files[idx + i]
        ref_fn, _, ref_albedo_fn, _, _, _  = ref_files[idx + i]

        hdr = load_exr(hdr_fn)
        normal = load_exr(normal_fn)[0:2, ...]
        albedo = load_exr(albedo_fn)
        ref = load_exr(ref_fn)
        ref_albedo = load_exr(ref_albedo_fn)

        concat = torch.cat([hdr, normal, albedo, ref, ref_albedo], dim=0)
        # hdr [0:3]
        # normal [3:5]
        # albedo [5:8]
        # ref [8:11]
        # ref_albedo [11:14]

        if args.extra:
            direct = load_exr(direct_fn)
            indirect = load_exr(indirect_fn)
            shadowt = load_exr(shadowt_fn)

            concat = torch.cat([concat, direct, indirect, shadowt], dim=0)

        temporal.append(concat)


    features = torch.stack(temporal)

    crops = get_crops(features)

    crop_filenames = []
    for crop_idx, crop in enumerate(crops):
        temporal_filenames = []
        for temporal_idx in range(args.num_temporal + 1):
            hdr_name = 'hdr_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            normal_name = 'normal_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            albedo_name = 'albedo_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            ref_name = 'ref_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            ref_albedo_name = 'ref_albedo_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            direct_name = 'direct_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            indirect_name = 'indirect_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            shadowt_name = 'shadowt_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)

            dump_raw(crop[temporal_idx, 0:3, ...].cpu(), os.path.join(dst_dir, hdr_name))
            dump_raw(crop[temporal_idx, 3:5, ...].cpu(), os.path.join(dst_dir, normal_name))
            dump_raw(crop[temporal_idx, 5:8, ...].cpu(), os.path.join(dst_dir, albedo_name))
            dump_raw(crop[temporal_idx, 8:11, ...].cpu(), os.path.join(dst_dir, ref_name))
            dump_raw(crop[temporal_idx, 11:14, ...].cpu(), os.path.join(dst_dir, ref_albedo_name))

            if args.extra:
                dump_raw(direct_crop[temporal_idx, 11:14, ...].cpu(), os.path.join(dst_dir, direct_name))
                dump_raw(indirect_crop[temporal_idx, 14:17, ...].cpu(), os.path.join(dst_dir, indirect_name))
                dump_raw(shadowt_crop[temporal_idx, 17:20, ...].cpu(), os.path.join(dst_dir, shadowt_name))

            temporal_filenames.append((hdr_name, normal_name, albedo_name, ref_name, ref_albedo_name, direct_name, indirect_name, shadowt_name))

        crop_filenames.append(temporal_filenames)
    filenames.append(crop_filenames)

metadata = {}
metadata['files'] = filenames
metadata['size'] = cropsize

with open(os.path.join(dst_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f)
