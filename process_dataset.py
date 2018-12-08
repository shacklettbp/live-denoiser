from dataset import get_files
from data_loading import load_exr, dump_raw
from utils import tonemap
import sys
import os
import torch
from itertools import chain

data_dir = sys.argv[1]
ref_dir = sys.argv[2]
dst_dir = sys.argv[3]
files = get_files(data_dir, 'exr', int(sys.argv[4]))
ref_files = get_files(ref_dir, 'exr', int(sys.argv[4]))
cropsize = 256

def get_crops(hdr, ref, normal, albedo):
    _, height, width = hdr.shape
    crops = []
    for y in chain(range(0, height, cropsize)[:-1], [height - cropsize]):
        for x in chain(range(0, width, cropsize)[:-1], [width - cropsize]):
            hdr_crop = hdr[:, y:y+cropsize, x:x+cropsize]
            ref_crop = ref[:, y:y+cropsize, x:x+cropsize]
            normal_crop = normal[:, y:y+cropsize, x:x+cropsize]
            albedo_crop = albedo[:, y:y+cropsize, x:x+cropsize]

            mse = ((torch.log1p(hdr_crop) - torch.log1p(ref_crop))**2).mean()
            crops.append((mse, hdr_crop, ref_crop, normal_crop, albedo_crop))

    crops = sorted(crops, key=lambda x: x[0])[::-1][:10]
    assert(len(crops) == 10)
    return crops

for idx, ((hdr, normal, albedo), (ref, _, _)) in enumerate(zip(files, ref_files)):
    hdr, normal, albedo, ref = load_exr(hdr).cuda(), load_exr(normal).cuda(), load_exr(albedo).cuda(), load_exr(ref).cuda()
    crops = get_crops(hdr, ref, normal, albedo)

    for crop_idx, (mse, hdr_crop, ref_crop, normal_crop, albedo_crop) in enumerate(crops):
        dump_raw(hdr_crop.cpu(), os.path.join(dst_dir, 'hdr_{}_{}.dmp'.format(idx, crop_idx)))
        dump_raw(ref_crop.cpu(), os.path.join(dst_dir, 'ref_{}_{}.dmp'.format(idx, crop_idx)))
        dump_raw(normal_crop.cpu(), os.path.join(dst_dir, 'normal_{}_{}.dmp'.format(idx, crop_idx)))
        dump_raw(albedo_crop.cpu(), os.path.join(dst_dir, 'albedo_{}_{}.dmp'.format(idx, crop_idx)))
