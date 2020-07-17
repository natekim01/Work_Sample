import argparse
import datetime
import errno
import os

def get_args():
  parser = argparse.ArgumentParser(description='dl-recon')
  now_str = datetime.datetime.now().strftime('%Y-%m-%d-%Hh-%Mm-%Ss')
  parser.add_argument('--batch_norm', type=int, default=1)
  parser.add_argument('--batch_size', type=int, default=48)
  parser.add_argument('--device_idx', type=int, default=1)
  parser.add_argument('--hidden_layer_width', type=int, default=1024)
  parser.add_argument('--sampling_pattern', type=str, default='kstride-2x2')
  parser.add_argument('--latest_model_dir', type=str,
                      default=os.path.join('/data/kimn/results', now_str))
  parser.add_argument('--latest_snapshots_dir', type=str,
                      default=os.path.join('/data/kimn/results',
                                           now_str, 'snapshots'))
  parser.add_argument('--learning_rate', type=float, default=1e-3)
  parser.add_argument('--loss', type=str, default='l2')
  parser.add_argument('--model_idx', type=int, default=4)
  parser.add_argument('--num_epochs', type=int, default=1000)
  parser.add_argument('--num_iters_per_eval', type=int, default=200)
  parser.add_argument('--num_hidden_fc_layers', type=int, default=1)
  parser.add_argument('--num_workers', type=int, default=1)
  parser.add_argument('--patch_sidelength_min', type=int, default=100)
  parser.add_argument('--patch_sidelength_max', type=int, default=300)
  parser.add_argument('--results_dir', type=str,
                      default='/data/kimn/results')
  parser.add_argument('--root_dir', type=str,
                      default='/data/kimn/bravo/norm-0')
  parser.add_argument('--save', type=int, default=1)
  parser.add_argument('--train', type=int, default=1)
  parser.add_argument('--val_fract', type=int, default=0.1)
  args = parser.parse_args()
  return args

# gateway fn to os.makedirs() that handles errors if directory already exists
def makedirs(name):
  try:
    os.makedirs(name)
  except OSError as ex:
    if ex.errno == errno.EEXIST and os.path.isdir(name):
      pass # ignore existing directory
    else:
      raise # a different error occurred
