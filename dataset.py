import torch.utils.data
import os
from glob import glob
from data_loading import pad_data, load_exr, load_raw
import numpy as np
import random

def get_files(dir, extension, num_imgs=None):
    files = []
    dir = os.path.expanduser(dir)
    if num_imgs is None:
        num_imgs = len(glob(os.path.join(dir, 'hdr*.{}'.format(extension))))
    for i in range(num_imgs):
        files.append((os.path.join(dir, "hdr_{}.{}".format(i, extension)),
                      os.path.join(dir, "normal_{}.{}".format(i, extension)),
                      os.path.join(dir, "albedo_{}.{}".format(i, extension)),
                      os.path.join(dir, "alt_hdr_{}.{}".format(i, extension))))

    return files

class DenoiserDataset(torch.utils.data.Dataset):
    def __init__(self, extension, training_path, reference_path=None, num_imgs=None, cropsize=(256, 256), augment=True):
        self.perform_augmentations = augment
        self.cropsize = cropsize
        self.training_files = get_files(training_path, extension, num_imgs=num_imgs)
        assert(len(self.training_files) > 0)
        if reference_path:
            self.reference_files = get_files(reference_path, extension, num_imgs=num_imgs)
            assert(len(self.reference_files) == len(self.training_files))

    def __len__(self):
        return len(self.training_files)

    def augment(self, color, normal, albedo, ref):
        if not self.perform_augmentations:
            if ref is not None:
                return [ color, normal, albedo, ref ]
            else:
                return [ color, normal, albedo ]

        assert(ref is not None)
        #if random.random() < 0.5:
        #    color = color.flip(-1)
        #    ref = ref.flip(-1)
        #    normal = normal.flip(-1)
        #    albedo = albedo.flip(-1)

        #color_indices = np.random.permutation(3)

        #color = color[:, color_indices, ...]
        #ref = ref[:, color_indices, ...]
        #albedo = albedo[:, color_indices, ...]

        return [ color, normal, albedo, ref ]

class ExrDataset(DenoiserDataset):
    def __init__(self, want_reference=True, *args, **kwargs):
        super(ExrDataset, self).__init__('exr', *args, **kwargs)
        self.want_reference = want_reference

    def __getitem__(self, idx):
        hdr_filename, normal_filename, albedo_filename, ref_hdr_filename = self.training_files[idx]

        color_tensor = load_exr(hdr_filename)
        normal_tensor = load_exr(normal_filename)
        albedo_tensor = load_exr(albedo_filename)

        _, height, width = color_tensor.shape

        max_top = height - self.cropsize[0]
        max_left = width - self.cropsize[1]

        col_idx = random.randint(0, max_left)
        row_idx = random.randint(0, max_top)

        color_tensor = color_tensor[...,
                                    row_idx:row_idx+self.cropsize[0],
                                    col_idx:col_idx+self.cropsize[1]]

        if self.want_reference:
            reference_tensor = load_exr(ref_hdr_filename)
            reference_tensor = reference_tensor[...,
                                                row_idx:row_idx+self.cropsize[0],
                                                col_idx:col_idx+self.cropsize[1]]
        else:
            reference_tensor = None

        normal_tensor = normal_tensor[...,
                                      row_idx:row_idx+self.cropsize[0],
                                      col_idx:col_idx+self.cropsize[1]]

        albedo_tensor = albedo_tensor[...,
                                      row_idx:row_idx+self.cropsize[0],
                                      col_idx:col_idx+self.cropsize[1]]

        return self.augment(color_tensor, normal_tensor, albedo_tensor, reference_tensor)

# FIXME out of date
class NumpyRawDataset(DenoiserDataset):
    def __init__(self, fullshape, *args, **kwargs):
        super(NumpyRawDataset, self).__init__('dmp', *args, **kwargs)
        self.fullshape = fullshape

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

class PreProcessedDataset(DenoiserDataset):
    def get_crop_files(self, dir, num_imgs):
        files = []
        dir = os.path.expanduser(dir)
        if num_imgs is None:
            num_imgs = len(glob(os.path.join(dir, 'hdr_*_0_0.dmp')))
        num_crops = len(glob(os.path.join(dir, 'hdr_0_*_0.dmp')))
        num_temporal = len(glob(os.path.join(dir, 'hdr_0_0_*.dmp')))
        for i in range(num_imgs):
            for crop in range(num_crops):
                temporal = []
                for temporal_idx in range(num_temporal):
                    temporal.append((os.path.join(dir, "hdr_{}_{}_{}.dmp".format(i, crop, temporal_idx)),
                              os.path.join(dir, "normal_{}_{}_{}.dmp".format(i, crop, temporal_idx)),
                              os.path.join(dir, "albedo_{}_{}_{}.dmp".format(i, crop, temporal_idx)),
                              os.path.join(dir, "ref_{}_{}_{}.dmp".format(i, crop, temporal_idx))))
                files.append(temporal)

        return files

    def __init__(self, dataset_path, num_imgs=None, size=(256, 256), augment=True):
        self.perform_augmentations = augment
        self.cropsize = size
        self.fullshape = (3, *size)
        self.training_files = self.get_crop_files(dataset_path, num_imgs)

    def __getitem__(self, idx):
        _, height, width = self.fullshape

        filenames = self.training_files[idx]
        color = [load_raw(f[0], self.fullshape) for f in filenames]
        normal = [load_raw(f[1], self.fullshape) for f in filenames]
        albedo = [load_raw(f[2], self.fullshape) for f in filenames]
        reference = [load_raw(f[3], self.fullshape) for f in filenames]

        color_tensor = torch.stack(color)
        normal_tensor = torch.stack(normal)
        albedo_tensor = torch.stack(albedo)
        reference_tensor = torch.stack(reference)

        return self.augment(color_tensor, normal_tensor, albedo_tensor, reference_tensor)
