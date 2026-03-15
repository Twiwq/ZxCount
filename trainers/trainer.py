import torch
import torch.nn as nn
import os
import re
import time
from glob import glob

from utils.misc import AverageMeter, DictAvgMeter, get_current_datetime, easy_track

class Trainer(object):
    def __init__(self, seed, version, device):

        self.seed = seed
        self.version = version
        self.device = torch.device(device)

        self.log_dir = os.path.join('logs', self.version)
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, msg, verbose=True, **kwargs):
        if verbose:
            print(msg, **kwargs)
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            if 'end' in kwargs:
                f.write(msg + kwargs['end'])
            else:
                f.write(msg + '\n')

    def load_ckpt(self, model, path):
        if path is not None:
            self.log('Loading checkpoint from {}'.format(path))
            model.load_state_dict(torch.load(path, map_location=self.device), strict=False)

    def save_ckpt(self, model, path):
        torch.save(model.state_dict(), path)

    def set_model_train(self, model):
        if isinstance(model, nn.Module):
            model.train()
        else:
            model[0].train()
            model[1].train()

    def set_model_eval(self, model):
        if isinstance(model, nn.Module):
            model.eval()
        else:
            model[0].eval()
            model[1].eval()

    def train_step(self, model, loss, optimizer, batch, epoch):
        pass

    def val_step(self, model, batch):
        pass

    def test_step(self, model, batch):
        pass

    def vis_step(self, model, batch):
        pass

  
    def test(self, model, test_dataloader, checkpoint_dir=None):
        self.log('Start testing at {}'.format(get_current_datetime()))

        if checkpoint_dir is None:
            checkpoint_dir = self.log_dir
        best_checkpoints = [f for f in os.listdir(checkpoint_dir) if re.match(r'.*best\d*\.pth', f)]

        if not best_checkpoints:
            self.log('No best checkpoints found in {}'.format(checkpoint_dir))
            return

        model = model.to(self.device) if isinstance(model, nn.Module) else [m.to(self.device) for m in model]

        for checkpoint in best_checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
            self.log('Testing with checkpoint: {}'.format(checkpoint_path))
            self.load_ckpt(model, checkpoint_path)

            self.set_model_eval(model)
            result_meter = DictAvgMeter()
            for batch in easy_track(test_dataloader, description='Testing with {}...'.format(checkpoint)):
                with torch.no_grad():
                    result = self.test_step(model, batch)
                result_meter.update(result)
            self.log('Testing results for {}:'.format(checkpoint), end=' ')
            for key, value in result_meter.avg.items():
                self.log('{}: {:.4f}'.format(key, value), end=' ')
            self.log('')

        self.log('Testing results saved to {}'.format(self.log_dir))
        self.log('End testing at {}'.format(get_current_datetime()))


    def vis(self, model, test_dataloader, checkpoint=None):
        self.log('Start visualization at {}'.format(get_current_datetime()))
        self.load_ckpt(model, checkpoint)

        os.makedirs(os.path.join(self.log_dir, 'vis'), exist_ok=True)

        model = model.to(self.device) if isinstance(model, nn.Module) else [m.to(self.device) for m in model]

        self.set_model_eval(model)
        for batch in easy_track(test_dataloader, description='Visualizing...'):
            with torch.no_grad():
                self.vis_step(model, batch)

        self.log('Visualization results saved to {}'.format(self.log_dir))
        self.log('End visualization at {}'.format(get_current_datetime()))
