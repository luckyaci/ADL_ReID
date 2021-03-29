from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import random
import numpy as np

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import opt

from io_stream import data_manager, NormalCollateFn, IdentitySampler
from utils.serialization import Logger, load_previous_model
from frameworks.models import ResNetBuilder
from frameworks.training import CameraClsTrainer, get_optimizer_strategy

from utils.serialization import Logger, save_checkpoint
from utils.transforms import TrainTransform


def train(**kwargs):
    opt._parse(kwargs)
    # torch.backends.cudnn.deterministic = True  # I think this line may slow down the training process
    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(os.path.join('./pytorch-ckpt/formalexp', opt.save_dir, 'log_train_camacc.txt'))

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(opt.trainset_name))
    train_dataset = data_manager.init_dataset(name=opt.trainset_name,
                                              num_bn_sample=opt.batch_num_bn_estimatation * opt.test_batch)
    pin_memory = True if use_gpu else False
    summary_writer = SummaryWriter(os.path.join('./pytorch-ckpt/formalexp', opt.save_dir, 'tensorboard_log'))

    trainloader = DataLoader(
        data_manager.init_datafolder(opt.trainset_name, train_dataset.gallery, TrainTransform(opt.height, opt.width)),
        sampler=IdentitySampler(train_dataset.train, opt.train_batch, opt.num_instances),
        batch_size=opt.train_batch, num_workers=opt.workers,
        pin_memory=pin_memory, drop_last=True, collate_fn=NormalCollateFn()
    )
    print('initializing model ...')
    model = ResNetBuilder(train_dataset.num_train_pids)
    #######test accuracy
    model_path = os.path.join("./pytorch-ckpt/formalexp", opt.save_dir,
                              'model_best.pth.tar')
    model = load_previous_model(model, model_path, load_fc_layers=False)
    model.eval()
    #######test accuracy
    optim_policy = model.get_optim_policy()
    print('model size: {:.5f}M'.format(sum(p.numel()
                                           for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()


    xent = nn.CrossEntropyLoss()

    def standard_cls_criterion(preditions,
                               targets,
                               global_step,
                               summary_writer):
        identity_loss = xent(preditions, targets)
        identity_accuracy = torch.mean((torch.argmax(preditions, dim=1) == targets).float())
        summary_writer.add_scalar('cls_loss', identity_loss.item(), global_step)
        summary_writer.add_scalar('cls_accuracy', identity_accuracy.item(), global_step)
        return identity_loss, identity_accuracy 

    # get trainer and evaluator
    optimizer, adjust_lr = get_optimizer_strategy(opt, optim_policy)
    reid_trainer = CameraClsTrainer(opt, model, optimizer, standard_cls_criterion, summary_writer)

    print('Start training')
    for epoch in range(opt.max_epoch):
        adjust_lr(optimizer, epoch)
        reid_trainer.train(epoch, trainloader)
        assert False

    if use_gpu:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    save_checkpoint({
        'state_dict': state_dict,
        'epoch': epoch + 1,
    }, save_dir=os.path.join('./pytorch-ckpt/formalexp', opt.save_dir))


if __name__ == '__main__':
    import fire
    fire.Fire()
