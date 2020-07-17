
import numpy as np
import pdb
import torch
import torch.nn.functional as f

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
  model = CNN0(input_shape, output_shape)

  # apply model to single-sample batch
  batch_input = sample['input'].unsqueeze(0)
  batch_output = model(batch_input)
  print('batch_output.shape is ' + str(batch_output.shape))

def main():
  test_class()

if __name__ == '__main__':
  main()
