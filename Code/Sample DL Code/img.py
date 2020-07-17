"""
img.py
implements methods for patch visualization

Nathanael Kim
"""

# load packages
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

# stitch together a list of possibly overlapping patches into a single image
#   patches are numpy arrays of arbitrary but equal shape
#   locations are int lists that mark where to place zeroth index of corresponding patches
def stitch(patches, locations, overlap_method='avg'):
  # choose method by which to resolve overlaps
  if overlap_method == 'avg':
    return stitch_by_averaging(patches, locations)
  else:
    raise ValueError('unknown overlap resolution method: %s.' % overlap_method)

# stitch patches by averaging across overlaps
def stitch_by_averaging(patches, locations):
  # find largest patch location index along each dimension for memory allocation
  dimension = len(locations[0])
  max_location = np.zeros(dimension, dtype='int')
  for location in locations:
    for i in range(dimension):
      if location[i] > max_location[i]:
        max_location[i] = location[i]

  # allocate memory for image and overlap counters at each image index
  patch_shape = np.array(patches[0].shape, dtype='int')
  image_shape = max_location + patch_shape
  image = np.zeros(image_shape, dtype=patches[0].dtype)
  overlap_counter = np.zeros(image_shape, dtype='int')

  # populate the image
  for patch, location in zip(patches, locations):
    # store indices corresponding to next patch location
    indices = []
    for index, patch_dim in zip(location, patch_shape):
      indices.append(slice(index, index + patch_dim))

    # accumulate patch into image
    image[tuple(indices)] += patch

    # accumulate overlap counter
    overlap_counter[tuple(indices)] += 1

  # compensate for overlapping by dividing image by number of overlaps
  # use a mask to only process indices touched by at least one patch
  mask = overlap_counter > 0
  image[mask] /= overlap_counter[mask]
  return image

# view three-dimensional image volume interactively
def view3d(volume, axis=0):
  print('view3d: use up/down arrows to scroll along axis %u' % axis)
  print('view3d: close figure to exit and proceed')

  # grab figure and axes objects
  fig, ax = plt.subplots()

  # set additional axes attributes
  ax.volume = volume
  ax.scroll_axis = axis
  ax.index = volume.shape[axis] // 2

  # show middle slice to start
  ax.imshow(volume.take(ax.index, axis=ax.scroll_axis),
            interpolation='nearest',
            cmap='gray')
  ax.set_yticks([])
  ax.set_xticks([])
  print_state(ax.index, volume.shape[ax.scroll_axis])

  # begin waiting for key_press events
  fig.canvas.mpl_connect('key_press_event', process_key)
  plt.show()

# process a key_press event by moving to an adjacent slice
def process_key(event):
  fig = event.canvas.figure
  ax = fig.axes[0]
  if event.key == 'up':
    show_previous_slice(ax)
  elif event.key == 'down':
    show_next_slice(ax)
  fig.canvas.draw()

# decrement the viewing slice number, wrapping around as needed
def show_previous_slice(ax):
  ax.index = (ax.index-1) % ax.volume.shape[ax.scroll_axis]
  ax.images[0].set_array(ax.volume.take(ax.index, axis=ax.scroll_axis))
  print_state(ax.index, ax.volume.shape[ax.scroll_axis])

# increment the viewing slice number, wrapping around as needed
def show_next_slice(ax):
  ax.index = (ax.index+1) % ax.volume.shape[ax.scroll_axis]
  ax.images[0].set_array(ax.volume.take(ax.index, axis=ax.scroll_axis))
  print_state(ax.index, ax.volume.shape[ax.scroll_axis])

def print_state(idx, max_idx):
  print('  showing slice %03u of %03u' % (idx, max_idx))

# print to file an image volume slice
def print_slice(volume, dim, slice, filename, ext='.eps'):
  fig, ax = plt.subplots()
  ax.imshow(volume.take(slice, axis=dim),
            interpolation='nearest',
            cmap='gray')
  ax.set_yticks([])
  ax.set_xticks([])
  fig.savefig(filename + ext, bbox_inches='tight', pad_inches=0)

# testbench: 2D simulation
def test_sim_2d():
  # define some test patches
  patches = []
  patch_size = [8, 6]
  patches.append(np.ones(patch_size) * 1.0)
  patches.append(np.ones(patch_size) * 2.0)
  patches.append(np.ones(patch_size) * 3.0)

  # define some test patch locations
  locations = [[0, 0], [2, 4], [6, 3]]

  # stitch the patches
  image = stitch(patches, locations)

  # display
  plt.imshow(image, interpolation='nearest', cmap='gray')
  plt.show()

# testbench: 3D case with interactive viewing
def test_sim_3d():
  # define some test patches
  patches = []
  patch_size = [2, 3, 4]
  patches.append(np.ones(patch_size) * 1.0)
  patches.append(np.ones(patch_size) * 2.0)
  patches.append(np.ones(patch_size) * 3.0)

  # define some test patch locations
  locations = [[0,0,0], [1,2,3], [2,4,0]]

  # stitch the patches
  image = stitch(patches, locations)

  # display
  view3d(image, axis=0)

# testbench: non-overlapping real data patches
def test_bravo_nooverlap():
  # set patch directory and extract patch size
  dir = '/data/kimn/bravo/norm-0/test/2019-02-06-12h/pass-1/targets,psize-1x206x152'
  psize = [int(i) for i in dir.split('-')[-1].split('x')]

  # set non-overlapping patch locations
  x_locations = np.arange(0, 80) * psize[0]
  y_locations = np.arange(0, 5) * psize[1]
  z_locations = np.arange(0, 5) * psize[2]

  # load patches and store their locations
  patches = []
  locations = []
  for x in x_locations:
    for y in y_locations:
      for z in z_locations:
        filename = os.path.join(dir, 'loc-{:03d}-{:03d}-{:03d}.bin'.format(x,y,z))
        patch = np.fromfile(filename, dtype='float32').reshape(psize, order='F')
        patches.append(patch)
        locations.append([x, y, z])

  # stitch the patches
  image = stitch(patches, locations)

  # display
  view3d(image, axis=0)

# testbench: overlapping real data patches
def test_bravo():
  # set patch directory and extract patch size
  dir = '/data/kimn/bravo/norm-0/test/2019-02-06-12h/pass-1/targets,psize-1x206x152'
  psize = [int(i) for i in dir.split('-')[-1].split('x')]

  # load patches and parse filenames to extract locations
  files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
  patches = []
  locations = []
  for file in files:
    patches.append(np.fromfile(os.path.join(dir, file), dtype='float32')
                   .reshape(psize, order='F'))
    locations.append([int(i) for i in file.split('.')[0].split('-')[1:]])

  # stitch the patches
  image = stitch(patches, locations)

  # display
  view3d(image, axis=0)

def main():
  test_bravo()

# execute main() only if module run directly (e.g., not imported)
if __name__ == '__main__':
  main()
