import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from opts import parse_opts
from model import generate_model

from dataset import get_data_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
from test import test_epoch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    opt = parse_opts()
    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if not opt.no_train:
        print('Setting up train_loader')
        training_data = get_data_set(opt, split='train_3d')
        train_loader = DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)


    if not opt.no_val:
        print('Setting up validation_loader')
        validation_data = get_data_set(opt, split='valid_3d')
        val_loader = DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')

    class EarlyStopping():
        def __init__(self):
            self.nsteps_similar_loss = 0
            self.previous_loss = 9999.0
            self.delta_loss = 0.01

        def _increment_step(self):
            self.nsteps_similar_loss += 1

        def _reset(self):
            self.nsteps_similar_loss = 0

        def eval_loss(self, loss):
            if (self.previous_loss - loss) <= self.delta_loss:
                self._increment_step()
                self.previous_loss = loss
            else:
                self._reset()
                self.previous_loss = loss

        def get_nsteps(self):
            return self.nsteps_similar_loss

    stop_criterion = EarlyStopping()

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

        stop_criterion.eval_loss(validation_loss)
        if stop_criterion.get_nsteps() >= 10:
            break

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

    if not opt.no_test:
        print('Setting up test_loader')
        test_data = get_data_set(opt, split='test_3d')
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        test_logger = Logger(
            os.path.join(opt.result_path, 'test.log'), ['loss', 'acc'])
        test_loss = test_epoch(test_loader, model, criterion, opt,
                               test_logger)
