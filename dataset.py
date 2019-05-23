import torch.utils.data
import os
from glob import glob
from data_loading import pad_data, load_exr, load_raw, dump_raw
import numpy as np
import random
import sys
import re
import json
from ast import literal_eval

def get_files(dir, extension, num_imgs=None, one_idx=False):
    files = []
    dir = os.path.expanduser(dir)
    if num_imgs is None:
        num_imgs = len(glob(os.path.join(dir, 'hdr*.{}'.format(extension))))

    start = 1 if one_idx else 0
    end = num_imgs + 1 if one_idx else num_imgs
    for i in range(start, end):
        files.append((os.path.join(dir, "hdr_{}.{}".format(i, extension)),
                      os.path.join(dir, "normal_{}.{}".format(i, extension)),
                      os.path.join(dir, "albedo_{}.{}".format(i, extension)),
                      os.path.join(dir, "direct_{}.{}".format(i, extension)),
                      os.path.join(dir, "indirect_{}.{}".format(i, extension)),
                      os.path.join(dir, "shadowt_{}.{}".format(i, extension))))

    return files

def augment_data(color, normal, albedo, ref, ref_albedo, direct, indirect, tshadow):
    assert(ref is not None)
    if random.random() < 0.5:
        color = color.flip(-1)
        normal = normal.flip(-1)
        albedo = albedo.flip(-1)
        ref = ref.flip(-1)
        ref_albedo = ref_albedo.flip(-1)
        direct = direct.flip(-1)
        indirect = indirect.flip(-1)
        tshadow = tshadow.flip(-1)

    color_indices = np.random.permutation(3)

    color = color[:, color_indices, ...]
    albedo = albedo[:, color_indices, ...]
    ref = ref[:, color_indices, ...]
    ref_albedo = ref_albedo[:, color_indices, ...]
    direct = direct[:, color_indices, ...]
    indirect = indirect[:, color_indices, ...]

    return [ color, normal, albedo, ref, ref_albedo, direct, indirect, tshadow ]

class ExrDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, num_imgs=None, cropsize=None):
        self.files = get_files(dataset_path, 'exr', num_imgs=num_imgs)
        self.cropsize = cropsize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hdr_filename, normal_filename, albedo_filename, direct_filename, indirect_filename, tshadow_filename = self.files[idx]

        color = load_exr(hdr_filename)
        normal = load_exr(normal_filename)[0:2, ...]
        albedo = load_exr(albedo_filename)
        #direct = load_exr(direct_filename)
        #indirect = load_exr(indirect_filename)
        #tshadow = load_exr(tshadow_filename)

        _, height, width = color.shape

        if self.cropsize is not None:
            max_top = height - self.cropsize[0]
            max_left = width - self.cropsize[1]

            col_idx = random.randint(0, max_left)
            row_idx = random.randint(0, max_top)

            color = color[...,
                          row_idx:row_idx+self.cropsize[0],
                          col_idx:col_idx+self.cropsize[1]]

            normal = normal[...,
                            row_idx:row_idx+self.cropsize[0],
                            col_idx:col_idx+self.cropsize[1]]

            albedo = albedo[...,
                            row_idx:row_idx+self.cropsize[0],
                            col_idx:col_idx+self.cropsize[1]]

        return [ color, normal, albedo]#, direct, indirect, tshadow ]

class ExrSeparatedDataset(torch.utils.data.Dataset):
    def get_files(self, dir, num_imgs, num_versions):
        files = []
        dir = os.path.expanduser(dir)
        if num_imgs is None:
            num_imgs = len(glob(os.path.join(dir, 'hdr_*_0.exr')))

        if num_versions is None:
            num_versions = len(glob(os.path.join(dir, 'hdr_0_*.exr')))

        for i in range(0, num_imgs):
            versions = []
            for j in range(num_versions):
                versions.append((os.path.join(dir, "hdr_{}_{}.exr".format(i, j)),
                              os.path.join(dir, "normal_{}_{}.exr".format(i, j)),
                              os.path.join(dir, "albedo_{}_{}.exr".format(i, j)),
                              os.path.join(dir, "direct_{}_{}.exr".format(i, j)),
                              os.path.join(dir, "indirect_{}_{}.exr".format(i, j)),
                              os.path.join(dir, "shadowt_{}_{}.exr".format(i, j))))

            files.append(versions)

        return files

    def __init__(self, dataset_path, num_imgs=None, num_versions=None):
        self.files = self.get_files(dataset_path, num_imgs=num_imgs, num_versions=num_versions)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        colors = []
        normals = []
        albedos = []
        for filenames in self.files[idx]:
            hdr_filename, normal_filename, albedo_filename, direct_filename, indirect_filename, tshadow_filename = filenames

            color = load_exr(hdr_filename)
            normal = load_exr(normal_filename)[0:2, ...]
            albedo = load_exr(albedo_filename)
            #direct = load_exr(direct_filename)
            #indirect = load_exr(indirect_filename)
            #tshadow = load_exr(tshadow_filename)

            colors.append(color)
            normals.append(normal)
            albedos.append(albedo)

        color = torch.stack(colors, dim=0)
        normal = torch.stack(normals, dim=0)
        albedo = torch.stack(albedos, dim=0)

        return [ color, normal, albedo]

class PreProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, augment=True, num_imgs=None):
        dataset_path = os.path.expanduser(dataset_path)

        with open(os.path.join(dataset_path, 'metadata.json')) as f:
            metadata = json.load(f)

        size = metadata['size']

        self.perform_augmentations = augment
        self.fullshape = (3, size[1], size[0])
        files = metadata['files']

        if num_imgs is None:
            num_imgs = len(files)

        for file_idx in range(num_imgs):
            crop_files = files[file_idx]
            for temporal_files in crop_files:
                for imgs in temporal_files:
                    for i, fname in enumerate(imgs):
                        imgs[i] = os.path.join(dataset_path, fname)

        self.files = files[0:num_imgs]

        self.need_pad = size[0] % 32 != 0 or size[1] % 32 != 0
        self.perform_augmentations = augment
        self.perform_augmentations = False

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        _, height, width = self.fullshape

        crops = self.files[idx]
        filenames = random.choice(crops)

        color = [load_raw(f[0], self.fullshape) for f in filenames]
        normal = [load_raw(f[1], (2, *self.fullshape[1:])) for f in filenames]
        albedo = [load_raw(f[2], self.fullshape) for f in filenames]
        reference = [load_raw(f[3], self.fullshape) for f in filenames]
        ref_albedo = [load_raw(f[4], self.fullshape) for f in filenames]
        #direct = [load_raw(f[4], self.fullshape) for f in filenames]
        #indirect = [load_raw(f[5], self.fullshape) for f in filenames]
        #tshadow = [load_raw(f[6], self.fullshape) for f in filenames]

        color = torch.stack(color)
        normal = torch.stack(normal)
        albedo = torch.stack(albedo)
        reference = torch.stack(reference)
        ref_albedo = torch.stack(ref_albedo)
        #direct = torch.stack(direct)
        #indirect = torch.stack(indirect)
        #tshadow = torch.stack(tshadow)
        #tshadow[torch.isnan(tshadow)] = 0
        #tshadow[torch.isinf(tshadow)] = 0

        if self.need_pad:
            color = pad_data(color)
            normal = pad_data(normal)
            albedo = pad_data(albedo)
            reference = pad_data(reference)
            ref_albedo = pad_data(ref_albedo)
            #direct = pad_data(direct)
            #indirect = pad_data(indirect)
            #tshadow = pad_data(tshadow)

        direct = None
        indirect = None
        tshadow = None

        if self.perform_augmentations:
            return augment_data(color, normal, albedo, reference, ref_albedo, direct, indirect, tshadow)
        else:
            return [color, normal, albedo, reference, ref_albedo]#, direct, indirect, tshadow]
