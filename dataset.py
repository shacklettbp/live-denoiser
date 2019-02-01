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
                      os.path.join(dir, "alt_hdr_{}.{}".format(i, extension))))

    return files

def augment_data(color, normal, albedo, ref):
    assert(ref is not None)
    if random.random() < 0.5:
        color = color.flip(-1)
        normal = normal.flip(-1)
        albedo = albedo.flip(-1)
        ref = ref.flip(-1)

    color_indices = np.random.permutation(3)

    color = color[:, color_indices, ...]
    albedo = albedo[:, color_indices, ...]
    ref = ref[:, color_indices, ...]

    return [ color, normal, albedo, ref ]

class ExrDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, training=True, num_imgs=None, cropsize=None):
        self.training = training 
        self.files = get_files(dataset_path, 'exr', num_imgs=num_imgs)
        self.cropsize = cropsize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hdr_filename, normal_filename, albedo_filename, ref_hdr_filename = self.files[idx]

        color = load_exr(hdr_filename)
        normal = load_exr(normal_filename)
        albedo = load_exr(albedo_filename)
        reference = load_exr(ref_hdr_filename) if self.training else None

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

            if self.training:
                reference = reference[...,
                                      row_idx:row_idx+self.cropsize[0],
                                      col_idx:col_idx+self.cropsize[1]]

        if self.training:
            return augment_data(color, normal, albedo, reference)
        else:
            return [ color, normal, albedo ]

# FIXME out of date
class NumpyRawDataset(torch.utils.data.Dataset):
    def __init__(self, fullshape, cropsize, num_imgs=None):
        self.files = get_files('dmp', num_imgs=num_imgs)
        self.fullshape = fullshape
        self.cropsize = cropsize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        _, height, width = self.fullshape

        max_top = height - self.cropsize[0]
        max_left = width - self.cropsize[1]

        col_idx = random.randint(0, max_left)
        row_idx = random.randint(0, max_top)

        hdr_filename, normal_filename, albedo_filename = self.training_files[idx]
        ref_hdr_filename, _, _ = self.reference_files[idx]

        color_tensor = load_raw_crop(hdr_filename, col_idx, row_idx, self.cropsize, self.fullshape)
        reference_tensor = load_raw_crop(ref_hdr_filename, col_idx, row_idx, self.cropsize, self.fullshape)
        normal_tensor = load_raw_crop(normal_filename, col_idx, row_idx, self.cropsize, self.fullshape)
        albedo_tensor = load_raw_crop(albedo_filename, col_idx, row_idx, self.cropsize, self.fullshape)

        return self.augment(color_tensor, reference_tensor, normal_tensor, albedo_tensor)

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

        self.files = files

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
        normal = [load_raw(f[1], self.fullshape) for f in filenames]
        albedo = [load_raw(f[2], self.fullshape) for f in filenames]
        reference = [load_raw(f[3], self.fullshape) for f in filenames]

        color_tensor = torch.stack(color)
        normal_tensor = torch.stack(normal)
        albedo_tensor = torch.stack(albedo)
        reference_tensor = torch.stack(reference)

        if self.need_pad:
            color_tensor = pad_data(color_tensor)
            normal_tensor = pad_data(normal_tensor)
            albedo_tensor = pad_data(albedo_tensor)
            reference_tensor = pad_data(reference_tensor)

        if self.perform_augmentations:
            return augment_data(color_tensor, normal_tensor, albedo_tensor, reference_tensor)
        else:
            return [color_tensor, normal_tensor, albedo_tensor, reference_tensor]
