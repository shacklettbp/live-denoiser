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
parser.add_argument('--extra', action='store_true', default=False)
args = parser.parse_args()

total_num_imgs = int(args.num_imgs)

num_imgs = total_num_imgs
if args.divide_data:
    num_imgs //= 2

data_dir = args.noisy
dst_dir = args.dst
files = get_files(data_dir, 'exr', total_num_imgs)

if args.single_reference:
    ref_file = args.ref
else:
    ref_files = get_files(args.ref, 'exr', num_imgs) if args.ref is not None else None

_, height, width = load_exr(files[0][0]).shape

if args.no_crop:
    cropsize = (width, height)
else:
    cropsize = (256, 256)

def get_crops(hdr, normal, albedo, ref, direct, indirect, shadowt):
    hdr[torch.isnan(hdr)] = 0 # NaNs
    ref[torch.isnan(ref)] = 0
    albedo[torch.isnan(albedo)] = 0
    normal[torch.isnan(normal)] = 0
    if args.extra:
        direct[torch.isnan(normal)] = 0
        indirect[torch.isnan(normal)] = 0
        shadowt[torch.isnan(normal)] = 0

    crops = []
    for y in chain(range(0, height, cropsize[1])[:-1], [height - cropsize[1]]):
        for x in chain(range(0, width, cropsize[0])[:-1], [width - cropsize[0]]):
            hdr_crop = hdr[0][:, y:y+cropsize[1], x:x+cropsize[0]]
            normal_crop = normal[0][:, y:y+cropsize[1], x:x+cropsize[0]]
            albedo_crop = albedo[0][:, y:y+cropsize[1], x:x+cropsize[0]]
            ref_crop = ref[0][:, y:y+cropsize[1], x:x+cropsize[0]]

            if args.extra:
                direct_crop = direct[0][:, y:y+cropsize[1], x:x+cropsize[0]]
                indirect_crop = indirect[0][:, y:y+cropsize[1], x:x+cropsize[0]]
                shadowt_crop = shadowt[0][:, y:y+cropsize[1], x:x+cropsize[0]]

            if torch.abs(hdr_crop - ref_crop).sum() == 0: # Skyboxes
                continue

            hdr_crop = [hdr_crop]
            normal_crop = [normal_crop]
            albedo_crop = [albedo_crop]
            ref_crop = [ref_crop]
            if args.extra:
                direct_crop = [direct_crop]
                indirect_crop = [indirect_crop]
                shadowt_crop = [shadowt_crop]
            else:
                direct_crop = None
                indirect_crop = None
                shadowt_crop = None

            for i in range(1, args.num_temporal+1):
                hdr_crop.append(hdr[i][:, y:y+cropsize[1], x:x+cropsize[0]])
                normal_crop.append(normal[i][:, y:y+cropsize[1], x:x+cropsize[0]])
                albedo_crop.append(albedo[i][:, y:y+cropsize[1], x:x+cropsize[0]])
                ref_crop.append(ref[i][:, y:y+cropsize[1], x:x+cropsize[0]])
                if args.extra:
                    direct_crop.append(direct[i][:, y:y+cropsize[1], x:x+cropsize[0]])
                    indirect_crop.append(indirect[i][:, y:y+cropsize[1], x:x+cropsize[0]])
                    shadowt_crop.append(shadowt[i][:, y:y+cropsize[1], x:x+cropsize[0]])

            crops.append((hdr_crop, normal_crop, albedo_crop, ref_crop, direct_crop, indirect_crop, shadowt_crop))

    return crops

filenames = []

for idx in range(0, num_imgs - args.num_temporal, args.num_temporal + 1):
    hdr = []
    normal = []
    albedo = []
    ref = []
    direct = []
    indirect = []
    shadowt = []

    for i in range(args.num_temporal + 1):
        hdr_fn, normal_fn, albedo_fn, ref_fn, direct_fn, indirect_fn, shadowt_fn = files[idx + i]
        if args.single_reference:
            ref_fn = ref_file
        elif ref_files is not None:
            ref_fn, _, _, _, _, _, _  = ref_files[idx + i]
        if args.divide_data:
            ref_fn, _, _, _, _, _, _ = files[num_imgs + idx + i]

        hdr.append(load_exr(hdr_fn))
        normal.append(load_exr(normal_fn)[0:2, ...])
        albedo.append(load_exr(albedo_fn))
        ref.append(load_exr(ref_fn))

        if args.extra:
            direct.append(load_exr(direct_fn))
            indirect.append(load_exr(indirect_fn))
            shadowt.append(load_exr(shadowt_fn))

    hdr = torch.stack(hdr)
    normal = torch.stack(normal)
    albedo = torch.stack(albedo)
    ref = torch.stack(ref)

    if args.extra:
        direct = torch.stack(direct)
        indirect = torch.stack(indirect)
        shadowt = torch.stack(shadowt)

    crops = get_crops(hdr, normal, albedo, ref, direct, indirect, shadowt)

    crop_filenames = []
    for crop_idx, (hdr_crop, normal_crop, albedo_crop, ref_crop, direct_crop, indirect_crop, shadowt_crop) in enumerate(crops):
        temporal_filenames = []
        for temporal_idx in range(args.num_temporal + 1):
            hdr_name = 'hdr_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            normal_name = 'normal_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            albedo_name = 'albedo_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            ref_name = 'ref_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            direct_name = 'direct_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            indirect_name = 'indirect_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)
            shadowt_name = 'shadowt_{}_{}_{}.dmp'.format(idx, crop_idx, temporal_idx)

            dump_raw(hdr_crop[temporal_idx].cpu(), os.path.join(dst_dir, hdr_name))
            dump_raw(normal_crop[temporal_idx].cpu(), os.path.join(dst_dir, normal_name))
            dump_raw(albedo_crop[temporal_idx].cpu(), os.path.join(dst_dir, albedo_name))
            dump_raw(ref_crop[temporal_idx].cpu(), os.path.join(dst_dir, ref_name))

            if args.extra:
                dump_raw(direct_crop[temporal_idx].cpu(), os.path.join(dst_dir, direct_name))
                dump_raw(indirect_crop[temporal_idx].cpu(), os.path.join(dst_dir, indirect_name))
                dump_raw(shadowt_crop[temporal_idx].cpu(), os.path.join(dst_dir, shadowt_name))

            temporal_filenames.append((hdr_name, normal_name, albedo_name, ref_name, direct_name, indirect_name, shadowt_name))

        crop_filenames.append(temporal_filenames)
    filenames.append(crop_filenames)

metadata = {}
metadata['files'] = filenames
metadata['size'] = cropsize

with open(os.path.join(dst_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f)
