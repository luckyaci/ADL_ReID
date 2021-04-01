from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import torch
from torch import nn
from frameworks.training.base import BaseTrainer
from utils.meters import AverageMeter
import time
from .loss import TripletLoss
import math
from .smooth import SmoothingForImage

class CameraClsTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion, summary_writer):
        super().__init__(opt, model, optimizer, criterion, summary_writer)
        self.camcri = nn.CrossEntropyLoss()
        self.triplet = TripletLoss(0.1, 'cosine')


    def _parse_data(self, inputs):
        imgs, pids, camids, path = inputs
        self.data = imgs.cuda()
        self.pids = pids.cuda()
        self.camids = camids.cuda()
        self.path = path



    def _forward(self, data, path):
        feat, id_scores, cam_scores, pzx_scores, pzpx_scores = self.model(data,path=path)
        return feat, id_scores, cam_scores, pzx_scores, pzpx_scores

    def _backward(self):
        self.loss.backward()

    def train(self, epoch, data_loader):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        all_camacc = 0
        all_idacc = 0
        for i, inputs in enumerate(data_loader):
            self._parse_data(inputs)


            torch.cuda.synchronize()
            tic = time.time()

            feat, id_scores, cam_scores, pzx_scores, pzpx_scores = self._forward(self.data, self.path)

            camids = self.camids
            pids = self.pids


            camloss = self.camcri(cam_scores, camids)
            camacc = torch.mean((torch.argmax(cam_scores, dim=1) == camids).float())
            idloss, identity_accuracy  = self.criterion(id_scores, pids, self.global_step,
                                       self.summary_writer)
            all_camacc = (i*all_camacc + camacc)/(i+1)
            all_idacc = (i*all_idacc + identity_accuracy)/(i+1)
            
            
            MI = -torch.mean(torch.log(pzx_scores)+torch.log(1-pzpx_scores))

            if i %20 == 0 :
                state = False
                set_train_mode(self.model, state)
                trainingDAL = not state
            elif i % 20 == 10:
                state = True
                set_train_mode(self.model, state)
                trainingDAL = not state


            if trainingDAL:
                self.loss =  idloss  + camloss*0.1  + MI			
                self.optimizer.zero_grad()
                self._backward()        
                self.optimizer.step()
            else:

                self.loss =  idloss  + camloss*0.1  + torch.exp(-MI/50)
                self.optimizer.zero_grad()
                self._backward()
                self.optimizer.step()

            losses.update(idloss)

            torch.cuda.synchronize()
            batch_time.update(time.time() - tic)

            self.global_step = epoch * len(data_loader) + i

            self.summary_writer.add_scalar('camloss', camloss.item(), self.global_step)
            self.summary_writer.add_scalar('camacc', camacc.item(), self.global_step)
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.summary_writer.add_scalar('MI', MI.item(), self.global_step)
            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'idLoss {:.3f} ({:.3f})\t'
                      'camLoss {:.3f}\t'
                      'MI {:.3f}\t'
                      'nan1 {:.3f}\t'
                      'nan2 {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.mean, batch_time.val,
                              losses.mean, losses.val,
                              camloss,
                              MI,
                              all_camacc,
                              all_idacc))

        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))


def set_train_mode(model, state):
	#print(state)
	set_grads(model.module.RFM, state)
	set_grads(model.module.base, state)
	#set_grads(model.module.bottleneck, state)
	set_grads(model.module.cam_classifier, state)
	set_grads(model.module.classifier, state)
	set_grads(model.module.DAL, not state)


def set_grads(mod, state):
	for name,para in mod.named_parameters():
		para.requires_grad = state


def flip_grads(mod):
	for name,para in mod.named_parameters():
		if para.requires_grad:
			para.grad = - para.grad


#cosine distance computation
def cosine(a,b):
    nor_a = a / a.norm(dim=-1, keepdim = True)
    nor_b = b / b.norm(dim=-1, keepdim = True)
    distance = torch.mm(nor_a ,nor_b.transpose(1,0))
    average_dis = distance.sum(dim=-1)/nor_b.size(0)
    average_dis = average_dis.sum()/nor_a.size(0)


    return torch.abs(average_dis)


