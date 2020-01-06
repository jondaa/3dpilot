"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random
import h5py
import torch
import data.transforms as transforms
from torch.utils.data import Dataset

import pytorch_nufft


class SliceData3D(Dataset):
    """
    A PyTorch Dataset that provides access to MR image 3d blocks.
    """

    def __init__(self, root, transform, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        self.transform = transform

        self.examples = []
        files = [x for x in pathlib.Path(root).iterdir() if not x.is_dir()]
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        self.examples=sorted(files)
        """
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]
            """

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace']
            target = data['reconstruction_esc'] if 'reconstruction_esc' in data else None
            return self.transform(kspace, target, data.attrs, fname.name)


class DataTransform3D:
    def __init__(self, resolution, depth,resolution_degrading):
        self.resolution = resolution
        self.depth = depth
        self.resolution_degrading=resolution_degrading

    def __call__(self, kspace, target, attrs, fname):


        target = transforms.to_tensor(target[:])
        target = torch.nn.functional.avg_pool2d(target,self.resolution_degrading)
        self.resolution=min(320//self.resolution_degrading,self.resolution)
        target = pytorch_nufft.transforms.center_crop_3d(target, (self.depth, self.resolution, self.resolution))
        target, mean, std = transforms.normalize_instance(target, eps=1e-11)
        target = target.clamp(-6, 6)
        kspace = pytorch_nufft.transforms.rfft3_regular(target)

        return kspace, target, mean, std