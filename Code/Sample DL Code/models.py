"""
neural network models

Nathanael Kim
"""

import numpy as np
import pdb
import torch
import torch.nn.functional as f

class Autoencoder0(torch.nn.Module):
  def __init__(self, input_shape, output_shape,
               hidden_layer_width=256,
               num_hidden_layers=2):
    super(Autoencoder0, self).__init__()

    # assign network shape parameters
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.num_hidden_layers = num_hidden_layers

    # define network layers
    self.fc_in  = torch.nn.Linear(np.prod(input_shape), hidden_layer_width)
    self.fc_hid = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
    self.fc_o   = torch.nn.Linear(hidden_layer_width, np.prod(output_shape))

  def forward(self, x):
    x = x.view(-1, np.prod(self.input_shape))
    x = self.fc_in(x)
    x = f.relu(x)
    for layer in range(self.num_hidden_layers):
      x = self.fc_hid(x)
      x = f.relu(x)
    x = self.fc_o(x)
    return x.view(-1, *self.output_shape)



class CNN0(torch.nn.Module):
  def __init__(self, input_shape, output_shape,
               batch_norm=False,
               num_hidden_fc_layers=1):
    super(CNN0, self).__init__()

    # store network shapes as attributes
    self.input_shape = input_shape
    self.output_shape = output_shape

    # define convolution-related hyperparameters
    conv3d_params = {
      'num_output_channels_layerwise': (64, 256, 1024),
      'dilation': (1, 1, 1),
      'kernel_shape': (1, 3, 3),
      'padding': (0, 0, 0),
      'stride': (1, 2, 2)
    }

    # define maxpool-related hyperparameters
    maxpool3d_params = {
      'dilation': (1, 1, 1),
      'kernel_shape': (1, 2, 2),
      'padding': (0, 0, 0),
      'stride': (1, 2, 2)
    }

    # initialize (pure-real) input channel dimension to 2*ncoil
    num_input_channels = np.prod(input_shape[3:])

    # keep track of current_shape for proper sizing of first hidden layer
    current_shape = [num_input_channels, *input_shape[0:3]]
    print('top: current_shape = %s' % str(current_shape))

    # define convolutional layers
    conv_layers = []
    for num_output_channels in conv3d_params['num_output_channels_layerwise']:
      # append a convolutional operation
      conv_layers.append(torch.nn.Conv3d(num_input_channels, num_output_channels,
                                         conv3d_params['kernel_shape'],
                                         stride=conv3d_params['stride'],
                                         padding=conv3d_params['padding'],
                                         dilation=conv3d_params['dilation']))
      current_shape = self.get_conv3d_output_shape(current_shape,
                                                   num_output_channels,
                                                   conv3d_params)
      print('after conv: current_shape = %s' % str(current_shape))

      # append a batch-normalization operation
      if batch_norm:
        conv_layers.append(torch.nn.BatchNorm3d(num_output_channels))

      # append a nonlinear activation function
      conv_layers.append(torch.nn.LeakyReLU())

      # append a max-pooling operation
      if current_shape[2]>1:
        conv_layers.append(torch.nn.MaxPool3d(maxpool3d_params['kernel_shape'],
                                              stride=maxpool3d_params['stride'],
                                              padding=maxpool3d_params['padding'],
                                              dilation=maxpool3d_params['dilation']))
        current_shape = self.get_maxpool3d_output_shape(current_shape,
                                                        maxpool3d_params)
        print('after max pool: current_shape = %s' % str(current_shape))

      # update num_input_channels
      num_input_channels = current_shape[0]

    # define fully-connected layers
    fc_layers = []
    fc_layer_width = np.prod(current_shape)
    for i in range(num_hidden_fc_layers):
      # append a hidden linear layer
      fc_layers.append(torch.nn.Linear(fc_layer_width, fc_layer_width))

      # append a batch-normalization operation
      if batch_norm:
        fc_layers.append(torch.nn.BatchNorm1d(fc_layer_width))

      # append a nonlinear activation function
      fc_layers.append(torch.nn.LeakyReLU())

    # define an output layer (no nonlinearity to retain full range)
    fc_layers.append(torch.nn.Linear(fc_layer_width, np.prod(output_shape)))

    # package conv and fc layers separately to enable intermediate reshaping
    self.conv_layers = torch.nn.Sequential(*conv_layers)
    self.fc_layers = torch.nn.Sequential(*fc_layers)

  def forward(self, x):
    x = x.view(-1, *self.input_shape[0:3], np.prod(self.input_shape[3:]))
    x = x.permute(0, 4, 1, 2, 3) # for compatibility with conv3d()
    x = self.conv_layers(x)
    x = x.view(-1, np.prod(x.shape[1:]))
    x = self.fc_layers(x)
    return x.view(-1, *self.output_shape)

  def get_conv3d_output_shape(self, current_shape,
                              num_output_channels, conv3d_params):
    return [num_output_channels,
            *self._get_output_shape_helper(current_shape, conv3d_params)]

  def get_maxpool3d_output_shape(self, current_shape, maxpool3d_params):
    return [current_shape[0],
            *self._get_output_shape_helper(current_shape, maxpool3d_params)]

  def _get_output_shape_helper(self, current_shape, params):
    # implements formulae given in pytorch documentation directly
    return [int(np.floor((current_shape[i+1]+2*params['padding'][i]-
                          params['dilation'][i]*(params['kernel_shape'][i]-1)-1)
                         / params['stride'][i] + 1))
            for i in range(3)]


class UNet0(torch.nn.Module):
  def __init__(self, input_shape, output_shape):
    super(UNet0, self).__init__()

    # store network shapes as attributes
    self.input_shape = input_shape
    self.output_shape = output_shape

    # initialize (pure-real) input channel dimension to 2*ncoil
    num_input_channels = np.prod(self.input_shape[3:])

    # store subnetwork instances
    self.inc = _unet_in(num_input_channels, 16)
    self.dwn0 = _unet_down(16, 32)
    self.dwn1 = _unet_down(32, 64)
    self.dwn2 = _unet_down(64, 128)
    self.dwn3 = _unet_down(128, 256)
    self.dwn4 = _unet_down(256, 512)
    self.dwn5 = _unet_down(512, 512, conv3d_kernel=(1, 2, 1))
    self.up0 = _unet_up(512, 256, conv3d_kernel=(1, 2, 1))
    self.up1 = _unet_up(256, 128)
    self.up2 = _unet_up(128, 64)
    self.up3 = _unet_up(64, 32)
    self.up4 = _unet_up(32, 16)
    self.up5 = _unet_up(16, 16)
    self.outc = _unet_out(16)

  def forward(self, x):
    x = x.view(-1, *self.input_shape[:3], np.prod(self.input_shape[3:]))
    x = x.permute(0, 4, 1, 2, 3) # for compatibility with conv3d()
    x0 = self.inc(x)
    x1 = self.dwn0(x0)
    x2 = self.dwn1(x1)
    x3 = self.dwn2(x2)
    x4 = self.dwn3(x3)
    x5 = self.dwn4(x4)
    x = self.dwn5(x5)
    x = self.up0(x, x5)
    x = self.up1(x, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    x = self.up5(x, x0)
    x = self.outc(x)
    zpad = [self.output_shape[i]-x.shape[i+2] for i in range(3)]
    x = f.pad(x, (zpad[2]//2, zpad[2]-zpad[2]//2,
                  zpad[1]//2, zpad[1]-zpad[1]//2,
                  zpad[0]//2, zpad[0]-zpad[0]//2))
    return x.view(-1, *self.output_shape)

class _unet_in(torch.nn.Module):
  def __init__(self, num_input_channels, num_output_channels,
               conv3d_kernel=(1, 3, 3),
               conv3d_stride=(1, 1, 1),
               conv3d_padding=(0, 0, 0),
               batch_norm=False):
    super(_unet_in, self).__init__()
    layers = [];
    layers.append(
      torch.nn.Conv3d(num_input_channels, num_output_channels, conv3d_kernel,
                      stride=conv3d_stride,
                      padding=conv3d_padding))
    if batch_norm:
      layers.append(torch.nn.BatchNorm3d(num_output_channels))
    layers.append(torch.nn.LeakyReLU(inplace=True))
    self.layers = torch.nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)

class _unet_down(torch.nn.Module):
  def __init__(self, num_input_channels, num_output_channels,
               maxpool3d_kernel=(1, 2, 2),
               maxpool3d_stride=(1, 2, 2),
               maxpool3d_padding=(0, 0, 0),
               conv3d_kernel=(1, 3, 3),
               conv3d_stride=(1, 1, 1),
               conv3d_padding=(0, 0, 0),
               batch_norm=False):
    super(_unet_down, self).__init__()
    layers = []
    layers.append(torch.nn.MaxPool3d(maxpool3d_kernel,
                                     stride=maxpool3d_stride,
                                     padding=maxpool3d_padding))
    layers.append(
      torch.nn.Conv3d(num_input_channels, num_output_channels, conv3d_kernel,
                      stride=conv3d_stride,
                      padding=conv3d_padding))
    if batch_norm:
      layers.append(torch.nn.BatchNorm3d(num_output_channels))
    layers.append(torch.nn.LeakyReLU(inplace=True))
    self.layers = torch.nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)

class _unet_up(torch.nn.Module):
  def __init__(self, num_input_channels, num_output_channels,
               convt3d_kernel=(1, 3, 3),
               convt3d_stride=(1, 1, 1),
               convt3d_padding=(0, 0, 0),
               upsamp_scale_factor=(1, 2, 2),
               upsamp_mode='trilinear',
               upsamp_align_corners=True,
               conv3d_kernel=(1, 3, 3),
               conv3d_stride=(1, 1, 1),
               conv3d_padding=(0, 0, 0),
               batch_norm=False):
    super(_unet_up, self).__init__()
    layers = []
    layers.append(
      torch.nn.ConvTranspose3d(num_input_channels, num_input_channels,
                               convt3d_kernel,
                               stride=convt3d_stride,
                               padding=convt3d_padding))
    layers.append(torch.nn.Upsample(scale_factor=upsamp_scale_factor,
                                    mode=upsamp_mode,
                                    align_corners=upsamp_align_corners))
    self.up = torch.nn.Sequential(*layers)
    layers = []
    layers.append(
      torch.nn.Conv3d(num_input_channels*2, num_output_channels, conv3d_kernel,
                      stride=conv3d_stride,
                      padding=conv3d_padding))
    if batch_norm:
      layers.append(torch.nn.BatchNorm3d(num_output_channels))
    layers.append(torch.nn.LeakyReLU(inplace=True))
    self.conv = torch.nn.Sequential(*layers)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = f.pad(x1, (0, x2.shape[4]-x1.shape[4],
                    0, x2.shape[3]-x1.shape[3],
                    0, x2.shape[2]-x1.shape[2]))
    return self.conv(torch.cat([x2, x1], dim=1))

class _unet_out(torch.nn.Module):
  def __init__(self, num_input_channels,
               conv3d_kernel=(1, 3, 3),
               conv3d_stride=(1, 1, 1),
               conv3d_padding=(0, 0, 0)):
    super(_unet_out, self).__init__()
    layers = []
    layers.append(
      torch.nn.Conv3d(num_input_channels, 1, conv3d_kernel,
                      stride=conv3d_stride,
                      padding=conv3d_padding))
    self.layers = torch.nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)

# unit test: class instantiation
def test_class():
  # load a sample
  import mridata
  dataset = mridata.Patches('/data/kimn/bravo/norm-0',
                            sampling_pattern='vdacc-4',
                            min_psize=[1, 100, 100],
                            max_psize=[1, 300, 300])
  index = 200
  sample = dataset[index]

  # instantiate a model
  input_shape = sample['input'].shape
  output_shape = sample['target'].shape
  model = UNet0(input_shape, output_shape)

  # apply model to single-sample batch
  batch_input = sample['input'].unsqueeze(0)
  batch_output = model(batch_input)
  print('batch_output.shape is ' + str(batch_output.shape))

def main():
  test_class()

if __name__ == '__main__':
  main()
