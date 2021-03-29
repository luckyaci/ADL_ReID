from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from frameworks.evaluating.base import BaseEvaluator


class FrameEvaluator(BaseEvaluator):
    def __init__(self, model, flip=True):
        super().__init__(model)
        self.loop = 2 if flip else 1
        self.evaluator_prints = []

    def _parse_data(self, inputs):
        imgs, pids, camids, path = inputs
        return imgs.cuda(), pids, camids

    def flip_tensor_lr(self, img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        if isinstance(feature, tuple) or isinstance(feature, list):
            output = []
            for x in feature:
                if isinstance(x, tuple) or isinstance(x, list):
                    output.append([item.cpu() for item in x])
                else:
                    output.append(x.cpu())
            return output
        else:
            return feature.cpu()

    def produce_features(self, dataloader, normalize=True):
        self.model.eval()
        all_feature_norm = []
        qf, q_pids, q_camids = [], [], []
        for batch_idx, inputs in enumerate(dataloader):
            inputs, pids, camids = self._parse_data(inputs)
            feature = None
            for i in range(self.loop):
                if i == 1:
                    inputs = self.flip_tensor_lr(inputs)
                global_f = self._forward(inputs)

                if feature is None:
                    feature = global_f
                else:
                    feature += global_f
            if normalize:
                fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
                all_feature_norm.extend(list(fnorm.cpu().numpy()[:, 0]))
                feature = feature.div(fnorm.expand_as(feature))
            else:
                feature = feature / 2
            qf.append(feature)
            q_pids.extend(pids)
            q_camids.extend(camids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        return qf, q_pids, q_camids

    def get_final_results_with_features(self, qf, q_pids, q_camids, gf, g_pids, g_camids, target_ranks=[1, 5, 10, 20]):
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

        dist_q_q = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                  torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dist_q_q.addmm_(1, -2, qf, qf.t())
        dist_q_q = dist_q_q.numpy()

        dist_g_g = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
        dist_g_g.addmm_(1, -2, gf, gf.t())
        dist_g_g = dist_g_g.numpy()

        """ print("starting rerank")
        distmat = re_ranking(distmat, dist_q_q, dist_g_g, k1=20, k2=6, lambda_value=0.3)
        print("finishing rerank") """


        """ qf = self.normalize(qf)
        #print(qf)
        gf = self.normalize(gf)
        #print(gf.shape)
        distmat = np.matmul(qf, gf.transpose()) """

        cmc, mAP, ranks = self.eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        self.evaluator_prints.append("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        self.evaluator_prints.append("CMC curve")
        for r in target_ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
            self.evaluator_prints.append("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
        return cmc[0]

    def collect_sim_bn_info(self, dataloader):
        network_bns = [x for x in list(self.model.modules()) if
                       isinstance(x, torch.nn.BatchNorm2d) or isinstance(x, torch.nn.BatchNorm1d)]
        for bn in network_bns:
            bn.running_mean = torch.zeros(bn.running_mean.size()).float().cuda()
            bn.running_var = torch.ones(bn.running_var.size()).float().cuda()
            bn.num_batches_tracked = torch.tensor(0).cuda().long()

        self.model.train()
        for batch_idx, inputs in enumerate(dataloader):
            # each camera should has at least 2 images for estimating BN statistics
            assert len(inputs[0].size()) == 4 and inputs[0].size(
                0) > 1, 'Cannot estimate BN statistics. Each camera should have at least 2 images'
            inputs, pids, camids = self._parse_data(inputs)
            for i in range(self.loop):
                if i == 1:
                    inputs = self.flip_tensor_lr(inputs)
                self._forward(inputs)
        self.model.eval()

    def normalize(self, x):
        norm = np.tile(np.sqrt(np.sum( np.square(x.numpy()), axis=1, keepdims=True)), [1, x.shape[1]])
        #print(norm)
        return x.numpy() / norm


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.4):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
    
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    print("coming soon")
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist