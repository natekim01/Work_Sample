"""
deep learning reconstruction of undersampled mri data

Nathanael Kim
"""

# built-in
import glob
import math
import os
import pdb
import time

# external
import numpy as np
import scipy.io
import torch

# this project
import autoencoders
import cnns
import img
import mridata
import unets
import utils

def train(dir, args):
  # set device to device_idx and get a pointer to it
  torch.cuda.set_device(args.device_idx)
  device = torch.device('cuda:{}'.format(args.device_idx))

  # instantiate training dataset class
  data = mridata.Patches(dir,
                         sampling_pattern=args.sampling_pattern,
                         min_psize=[1]+[args.patch_sidelength_min]*2,
                         max_psize=[1]+[args.patch_sidelength_max]*2)
  input_shape = data.get_padded_input_shape()
  target_shape = data.get_padded_target_shape()

  # instantiate training and validation data loaders
  # sample data separately for training versus validation
  indices = np.arange(len(data))
  np.random.shuffle(indices)
  split_index = int(np.floor(args.val_fract * len(data)))
  train_indices, val_indices = indices[split_index:], indices[:split_index]
  train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
  train_loader = torch.utils.data.DataLoader(data,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             sampler=train_sampler)
  val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
  val_loader = torch.utils.data.DataLoader(data,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           sampler=val_sampler)
  print('training with %u samples...' % len(train_indices))
  print('validating with %u samples...' % len(val_indices))

  # instantiate network
  if args.model_idx == 0:
    model = autoencoders.Autoencoder0(input_shape, target_shape,
                                      hidden_layer_width=args.hidden_layer_width,
                                      num_hidden_layers=args.num_hidden_fc_layers)
    print('using autoencoder0 model...')
  elif args.model_idx == 1:
    model = autoencoders.Autoencoder1(input_shape, target_shape,
                                      hidden_layer_width=args.hidden_layer_width,
                                      num_hidden_layers=args.num_hidden_fc_layers)
    print('using autoencoder1 model...')
  elif args.model_idx == 2:
    model = cnns.CNN0(input_shape, target_shape,
                      batch_norm=bool(args.batch_norm),
                      num_hidden_fc_layers=args.num_hidden_fc_layers)
    print('using cnn0 model...')
  elif args.model_idx == 3:
    model = unets.UNet0(input_shape, target_shape,
                        batch_norm=bool(args.batch_norm))
    print('using unet0 model...')
  elif args.model_idx == 4:
    model = unets.UNet1(input_shape, target_shape,
                        batch_norm=bool(args.batch_norm))
    print('using unet1 model...')
  else:
    error('unknown model_idx')
  model = model.cuda()

  # instantiate training loss
  if args.loss == 'l2':
    loss = torch.nn.MSELoss()
    print('using l2 loss...')
  elif args.loss == 'l1':
    loss = torch.nn.L1Loss()
    print('using l1 loss...')
  else:
    error('unknown loss')
  loss = loss.cuda()

  # instantiate optimizer
  optim = torch.optim.Adam(model.parameters(),
                           lr=args.learning_rate)
  print('using adam optimizer...')

  # setup progress updates
  headers = ['time', 'iter', 'epoch', 'batch_idx', 'batch_loss', 'val_loss']
  header_str = '{:>12}'*4 + '{:>16}'*2
  print(header_str.format(*headers))
  row_fmt = '{:>12.3f}' + '{:>12d}'*3 + '{:>16.4f}'*2
  val_fmt = 'epoch-{:06d},batch_idx-{:03d},batch_loss-{:7e},val_loss-{:7e}.pt'

  # create directories to save results if needed
  if args.save:
    utils.makedirs(args.latest_model_dir)
    utils.makedirs(args.latest_snapshots_dir)

  # initialize iteration count, best validation loss, training start time
  iter = 0
  start = time.time()
  best_filename = ''
  best_val_loss = math.inf

  # set model to training mode
  model.train()

  for epoch in range(args.num_epochs):
    for batch_idx, batch_data in enumerate(train_loader):
      iter += 1

      # forward pass
      batch_output = model(batch_data['input'].cuda())

      # compute loss
      batch_loss = loss(batch_output, batch_data['target'].cuda())

      # update network parameters via backpropagation
      optim.zero_grad()
      batch_loss.backward()
      optim.step()

      # evaluate performance on validation set periodically
      if (iter % args.num_iters_per_eval) == 0:
        # switch model to evaluation mode
        model.eval()

        # accumulate validation loss on current model
        val_loss = 0
        with torch.no_grad():
          for val_batch_idx, val_batch_data in enumerate(val_loader):
            val_batch_output = model(val_batch_data['input'].cuda())
            val_loss += loss(val_batch_data['target'].cuda(),
                             val_batch_output) * val_batch_output.shape[0]
          val_loss /= len(val_indices)

        # print progress update
        print(row_fmt.format(time.time()-start, iter, epoch, batch_idx,
                             batch_loss.item(), val_loss.item()))

        if args.save:
          # save this intermediate model
          val_filename = val_fmt.format(epoch, batch_idx,
                                        batch_loss.item(), val_loss.item())
          torch.save(model,
                     os.path.join(args.latest_snapshots_dir, val_filename))

          # update 'best' model if validation loss reduced
          if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_filename = val_filename

        # switch model back to training mode
        model.train()

  # copy the best model into args.latest_model_dir and return its full pathname
  if args.save:
    best_fullname = os.path.join(args.latest_snapshots_dir, best_filename)
    os.system('cp {} {}'.format(best_fullname, args.latest_model_dir))
    return os.path.join(args.latest_model_dir, best_filename)
  else:
    return None

# apply a model to test data
def test(dir, model_fullpath, args):
  with torch.no_grad():
    # set device to device_idx
    torch.cuda.set_device(args.device_idx)
    device = torch.device('cuda:{}'.format(args.device_idx))

    # load model into a function handle
    model = torch.load(model_fullpath)

    # set model to appropriate mode
    #   for models that use batch normalization, need to use model.train()
    #   for other models, can use model.eval() as expected
    if bool(args.batch_norm):
      model.train()
    else:
      model.eval()

    # instantiate test dataset class
    data = mridata.Patches(dir,
                           padded_input_shape=model.input_shape,
                           sampling_pattern=args.sampling_pattern,
                           min_psize=[1]+[args.patch_sidelength_min]*2,
                           max_psize=[1]+[args.patch_sidelength_max]*2)

    # instantiate test data loader
    indices = np.arange(len(data))
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    loader = torch.utils.data.DataLoader(data,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         sampler=sampler)
    print('evaluating on %u samples...' % len(data))

    # apply model batchwise on test data
    outputs = []
    targets = []
    locations = []
    for batch_idx, batch_data in enumerate(loader):
      # forward pass
      batch_output = model(batch_data['input'].cuda()).cpu()
      # todo: avoid moving data to device_idx and back

      # append each batch element to lists for output
      for i in range(batch_data['input'].shape[0]):
        outputs.append(batch_output[i,...].numpy())
        targets.append(batch_data['target'][i,...].numpy())
        locations.append(batch_data['target_location'][i,...].tolist())

    return outputs, targets, locations

# starting point if module run directly
def main():
  # set optional arguments
  args = utils.get_args()

  # train the model unless specified otherwise
  if args.train:
    train_dir = args.root_dir + '/train'
    model_fullpath = train(train_dir, args)
    if model_fullpath == None:
      print('exiting without evaluating since model not saved')
      return
  else:
    # use the most recently trained model
    model_fullpath = glob.glob(os.path.join(args.results_dir,
                                            max(os.listdir(args.results_dir)),
                                            '*.pt'))[0]

  # load and apply model on test data
  # str = '2019-07-19-07h,post'
  # str = '2019-07-22-14h,post'
  # str = '2019-07-24-17h,pre'
  # str = '2019-10-30-20h,post'
  # str = '2019-11-01-12h,post'
  str = '2019-11-01-12h,post'
  test_dir = args.root_dir + '/test/' + str
  t = time.time()
  outputs, targets, locations = test(test_dir, model_fullpath, args)
  print('recon done in %0.5f seconds.' % (time.time()-t))

  # stitch the patches (assumes single dataset)
  # todo: handle multiple-dataset case
  output_image = img.stitch(outputs, locations)
  target_image = img.stitch(targets, locations)

  # save results
  scipy.io.savemat('results,dl.mat',
                   dict(xhat=output_image, x=target_image))

  # interactive display
  concat_axis = 1
  scroll_axis = 2
  img.view3d(
    np.concatenate((output_image, target_image), axis=concat_axis),
    axis=scroll_axis)

# boilerplate
if __name__ == '__main__':
  main()
