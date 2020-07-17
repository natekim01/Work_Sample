"""
mridata.py
defines mri dataset classes and provides methods for initialization and loading

Nathanael Kim
"""

# load packages
import glob
import numpy as np
import os
import pandas as pd
import pdb
import random
import skimage.transform
import time
import torch
import torch.utils.data

# define dataset class
# inherit abstract class Dataset and override __init__, __len__, __getitem__
class Patches(torch.utils.data.Dataset):
  # initializes class instance
  def __init__(self, root_dir, padded_input_shape=[],
               sampling_pattern='kstride-2x1',
               min_psize=[1, 100, 100],
               max_psize=[1, 300, 300]):
    self.root_dir = root_dir
    self.meta = self._load_metadata(sampling_pattern, min_psize, max_psize)
    self.nsamp = len(self.meta)
    self._set_padded_shapes(padded_input_shape)

  # returns total number of samples
  def __len__(self):
    return self.nsamp

  # get (source, target) training sample as a dict
  def __getitem__(self, idx):
    # load binary files into numpy arrays
    input = np.fromfile(self.meta[idx][0], dtype='float32')
    target = np.fromfile(self.meta[idx][1], dtype='float32')

    # reshape to multidimensional arrays
    #   use 'F'ortran-like indexing since files were written in matlab
    #   input size: [psize_x, psize_y, psize_z, ncc, 2]
    #   target size: [psize_x, psize_y, psize_z]
    input = input.reshape(self.meta[idx][2], order='F')
    target = target.reshape(self.meta[idx][3], order='F')

    # zero-pad so spatial dims are centered, rx coil dim is left-adjusted
    zpad = [self.padded_input_shape[i]-input.shape[i]
            for i in range(len(input.shape))]
    zpad = ((zpad[0]//2, zpad[0]-zpad[0]//2),
            (zpad[1]//2, zpad[1]-zpad[1]//2),
            (zpad[2]//2, zpad[2]-zpad[2]//2),
            (0, zpad[3]),
            (0, zpad[4]))
    input = np.pad(input, zpad, 'constant')
    target = np.pad(target, zpad[0:3], 'constant')

    """
    # omitting resampling for now since __getitem__ takes ~16x longer with it
    # zero-pad rx coil dimension to maximum
    zpad = self.padded_input_shape[3]-input.shape[3]
    input = np.pad(input, ((0,0), (0,0), (0,0), (0,zpad), (0,0)), 'constant')

    # resample to standardize matrix size
    input = skimage.transform.resize(input, self.padded_input_shape,
                                     mode='constant',
                                     preserve_range='True',
                                     anti_aliasing='False').astype('float32')

    target = skimage.transform.resize(target, self.padded_target_shape,
                                      mode='constant',
                                      preserve_range='True',
                                      anti_aliasing='False').astype('float32')
    """

    # transform to pytorch tensors and package for output
    sample = {'input': torch.tensor(input),
              'target': torch.tensor(target),
              'input_location': torch.tensor(self.meta[idx][4], dtype=torch.int),
              'target_location': torch.tensor(self.meta[idx][5], dtype=torch.int)}
    return sample

  # load metadata for this class instance
  def _load_metadata(self, sampling_pattern, min_psize, max_psize):
    # iterate over datasets, processing inputs and targets jointly
    metadata = []
    for elem in os.walk(self.root_dir):
      if sampling_pattern in elem[0] and 'stat' not in elem[0]:
        # parse subdirectory name to extract patch dimensions
        input_dir = elem[0]
        input_dims = [int(i) for i in input_dir.split('-')[-1].split('x')] + [2]

        # process subdirectory only for patch sizes within range of interest
        if all(min_psize[i]<=input_dims[i] and input_dims[i]<=max_psize[i]
               for i in range(len(min_psize))):
          # get corresponding target subdirectory
          target_dims = input_dims[0:3]
          psize_str = 'psize-' + 'x'.join(str(i) for i in target_dims)
          target_dir = glob.glob(os.path.join(elem[0],'..',
                                              'targets,*' + psize_str + '*'))[0]

          # iterate over current dataset's input files
          for input_filename in elem[2]:
            # target file is named the same as input file, except no coil index
            target_filename = input_filename[:-8] + '.bin'

            # define full filenames
            input_fullname = os.path.join(input_dir, input_filename)
            target_fullname = os.path.join(target_dir, target_filename)

            # parse input and target filenames to extract patch locations
            input_loc = [int(i) for i in input_filename.split('.')[0].split('-')[1:]]
            target_loc = [int(i) for i in target_filename.split('.')[0].split('-')[1:]]

            # append metadata
            metadata.append((input_fullname, target_fullname,
                             input_dims, target_dims,
                             input_loc, target_loc))
    return metadata

  # sets padded input shape
  def _set_padded_shapes(self, padded_input_shape):
    if padded_input_shape:
      self.padded_input_shape = padded_input_shape
    else:
      self.padded_input_shape = [0, 0, 0, 0, 0]
      for item in self.meta:
        self.padded_input_shape = [max(self.padded_input_shape[i], item[2][i])
                                   for i in range(len(item[2]))]
    self.padded_target_shape = self.padded_input_shape[0:3]

  # returns padded input shape
  def get_padded_input_shape(self):
    return self.padded_input_shape

  # returns padded target shape
  def get_padded_target_shape(self):
    return self.padded_target_shape

# testbench
def main():
  # create class instance
  bravo = Patches(root_dir='/data/kimn/bravo/norm-0/train',
                  sampling_pattern='vdacc-4',
                  min_psize=[1, 100, 100],
                  max_psize=[1, 300, 300])

  # test Patches.__len__()
  nsamp = len(bravo)
  print('bravo dataset contains %u samples' % nsamp)

  # test Patches.get_padded_input_shape()
  padded_input_shape = bravo.get_padded_input_shape()
  print('bravo padded input shape is ' + str(padded_input_shape))

  # test Patches.get_padded_target_shape()
  padded_target_shape = bravo.get_padded_target_shape()
  print('bravo padded target shape is ' + str(padded_target_shape))

  # test Patches.__getitem__() on a randomly selected item
  idx = random.randint(0, nsamp)
  t = time.time()
  sample = bravo[idx]
  print('grabbed sample %u in %0.5f seconds.' % (idx, time.time()-t))
  print('  input is of shape ' + str(sample['input'].shape))
  print('  target is of shape ' + str(sample['target'].shape))
  print('  input location is ' + str(sample['input_location']))
  print('  target location is ' + str(sample['target_location']))

# execute main() only if module run directly (e.g., not imported)
if __name__ == '__main__':
  main()
