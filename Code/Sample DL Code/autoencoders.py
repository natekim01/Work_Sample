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


class Autoencoder1(torch.nn.Module):
  def __init__(self, input_shape, output_shape,
               hidden_layer_width=256,
               num_hidden_layers=2):
    super(Autoencoder1, self).__init__()

    # assign network shape parameters
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.num_hidden_layers = num_hidden_layers

    # define network layers
    self.fc_in  = torch.nn.Linear(np.prod(input_shape), hidden_layer_width)
    self.bn     = torch.nn.BatchNorm1d(hidden_layer_width)
    self.fc_hid = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
    self.fc_o   = torch.nn.Linear(hidden_layer_width, np.prod(output_shape))

  def forward(self, x):
    x = x.view(-1, np.prod(self.input_shape))
    x = self.fc_in(x)
    x = f.leaky_relu(x)
    x = self.bn(x)
    for layer in range(self.num_hidden_layers):
      x = self.fc_hid(x)
      x = f.leaky_relu(x)
      x = self.bn(x)
    x = self.fc_o(x)
    return x.view(-1, *self.output_shape)

# unit test: class instantiation
def test_class():
  # load a sample
  import mridata
  dataset = mridata.Patches('/data/kimn/bravo/norm-0',
                            sampling_pattern='kstride-2x1',
                            min_psize=[1, 64, 64],
                            max_psize=[1, 64, 64])
  index = 200
  sample = dataset[index]

  # instantiate a model
  input_shape = sample['input'].shape
  output_shape = sample['target'].shape
  model = Autoencoder0(input_shape, output_shape)

  # apply model to single-sample batch
  batch_input = sample['input'].unsqueeze(0)
  batch_output = model(batch_input)
  print('batch_output.shape is ' + str(batch_output.shape))

def main():
  test_class()

if __name__ == '__main__':
  main()
