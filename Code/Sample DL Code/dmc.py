"""
dcm.py
implements methods for handling dicom files

Nathanael Kim
memorial sloan kettering cancer center
"""

# load packages
import datetime
import img
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pydicom
import time
import utils

# write results alongside dicoms to file
def pretty_print(dir, dcmpath, sl, beta, crop, filename):
  # load results
  output = np.load(os.path.join(os.getcwd(), 'npy', dir + ',output.npy'))
  target = np.load(os.path.join(os.getcwd(), 'npy', dir + ',target.npy'))

  # postprocess results by cropping and low-pass filtering
  output_post = window(output.take(sl, axis=2)[:,:256], beta)
  output_post = rescale(output_post)
  target_post = window(target.take(sl, axis=2)[:,:256], beta)
  target_post = rescale(target_post)

  # load product dicom
  dcm = pydicom.dcmread(dcmpath)

  # postprocess product dicoms to match results
  dcm_img = dcm.pixel_array.astype('float32')
  dcm_img = rescale(dcm_img)

  # interactive display
  fig, ax = plt.subplots()
  ax.imshow(np.concatenate((target_post[crop],
                            dcm_img[crop],
                            output_post[crop]), axis=1),
            interpolation='nearest',
            cmap='gray')
  ax.set_yticks([])
  ax.set_xticks([])
  fig.savefig(filename, bbox_inches='tight', pad_inches=0)
  plt.show()

# filter 2d image with kaiser-bessel window
def window(img, beta):
  win = np.outer(np.kaiser(img.shape[0], beta),
                 np.kaiser(img.shape[1], beta))
  img = np.fft.fftshift(img)
  img = np.fft.fft2(img)
  img = np.fft.ifftshift(img)
  img = win * img
  img = np.fft.ifftshift(img)
  img = np.fft.ifft2(img)
  img = np.fft.fftshift(img)
  return np.abs(img)

# rescale
def rescale(img):
  img -= np.min(img)
  img /= np.max(img)
  return img

# write dicoms (not working)
def write_dicom():
  # load results
  str = '2019-07-22-14h,post'
  output_image = np.load(os.path.join(os.getcwd(), 'npy', str + ',output.npy'))

  # convert results into int16
  output_image -= np.min(output_image)
  output_image /= np.max(output_image)
  output_image *= 255
  output_image = output_image.astype('uint16')

  # write dicoms
  dicom_dir = '/Volumes/MRIClinical/Ricardo/BRAVO-newcases'
  exam_dir = '38069839/33224249/D2019_07_22/MR'
  series_dir = 'S0011_Axial BRAVO x4 acc'
  utils.makedirs(os.path.join(dicom_dir, exam_dir, series_dir))
  for i in range(output_image.shape[2]):
    file = 'ax-2,sl-{:03d}.dcm'.format(i)
    write_dicom(output_image[:,:,i],
                os.path.join(dicom_dir, exam_dir, series_dir, file))
  return

  # load one dicom
  dicom_dir = '/Volumes/MRIClinical/Ricardo/BRAVO-newcases'
  exam_dir = '38069839/33224249/D2019_07_22/MR'
  series_dir = 'S0011_Axial BRAVO'
  file = 'img_0001_2.789.dcm'
  ds = pydicom.dcmread(os.path.join(dicom_dir, exam_dir, series_dir, file))

  # modify metadata
  ds.Columns = output_image.shape[0]
  ds.Rows = output_image.shape[1]
  ds.SeriesDescription = u'Axial BRAVO x4 acc'

  # write new dicoms
  series_dir = 'S0012_Axial BRAVO 4x-acc'
  utils.makedirs(os.path.join(dicom_dir, exam_dir, series_dir))
  for i in range(output_image.shape[2]):
    ds.PixelData = output_image[:,:,i].tostring()
    file = 'ax-2,sl-{:03d}.dcm'.format(i)
    pydicom.dcmwrite(os.path.join(dicom_dir, exam_dir, series_dir, file), ds,
                     write_like_original=False)

# testbench
def main():
  results_dir = '2019-07-19-07h,post'
  dicompath = os.path.join('/Volumes/MRIClinical/Ricardo/BRAVO-newcases',
                           '38071536/33234067/D2019_07_19/MR',
                           'S0010_Axial BRAVO post',
                           'img_0111_2.455.dcm')
  sl = 119
  beta = 2
  crop = (slice(20, 240), slice(35, 215))
  filename = results_dir + ',ax-2,sl-{:03d}.eps'.format(sl)
  pretty_print(results_dir, dicompath, sl, beta, crop, filename)

  results_dir = '2019-07-22-14h,post'
  dicompath = os.path.join('/Volumes/MRIClinical/Ricardo/BRAVO-newcases',
                           '38069839/33224249/D2019_07_22/MR',
                           'S0010_Axial BRAVO post',
                           'img_0090_2.565.dcm')
  sl = 101
  crop = (slice(30, 250), slice(30, 220))
  filename = results_dir + ',ax-2,sl-{:03d}.eps'.format(sl)
  pretty_print(results_dir, dicompath, sl, beta, crop, filename)

if __name__ == '__main__':
  main()
