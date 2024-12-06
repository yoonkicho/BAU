  
from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluation_metrics import accuracy
from .loss import CrossEntropyLabelSmooth, TripletLoss

from .utils.meters import AverageMeter
from torch.cuda.amp import GradScaler, autocast


class BAUTrainer(object):
    def __init__(self, model, memory_bank, num_classes, margin, lam=1.5, k=10):
        super(BAUTrainer, self).__init__()
        self.model = model
        self.memory_bank = memory_bank
        
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.scaler = GradScaler()

        self.lam = lam
        self.k = k

    def train(self, epoch, train_loader, optimizer, iters, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_align = AverageMeter()
        losses_uniform = AverageMeter()
        losses_domain = AverageMeter()
        precisions = AverageMeter()

        time.sleep(1)
        end = time.time()
        for i in range(iters):
            batch_data = train_loader.next()
            inputs_w, inputs_s, pids, dids = self._parse_data(batch_data)
            
            optimizer.zero_grad()
            with autocast():
                # feedforward
                inputs = torch.cat([inputs_w, inputs_s], dim=0)
                emb, f, logits = self.model(inputs)
                f = F.normalize(f)
                emb_w, _ = emb.chunk(2)
                f_w, f_s = f.chunk(2)
                logits_w, _ = logits.chunk(2)

                # compute weights
                with torch.no_grad():
                    sims = torch.matmul(f, f.t()) # b*b
                    topk = torch.sort(sims, dim=1, descending=True).indices[:, :self.k] # b*k
                    nn = torch.zeros_like(sims).cuda()
                    rows = torch.arange(nn.size(0)).unsqueeze(1).expand_as(topk)
                    nn[rows, topk] = 1
                    reciprocal = nn * nn.t() # b*b

                    reciprocal_w, reciprocal_s = reciprocal.chunk(2)

                    intersection = torch.matmul(reciprocal_s, reciprocal_w.t()) # b*b
                    union = reciprocal_s.sum(1,keepdim=True) + reciprocal_w.sum(1,keepdim=True).t() - intersection
                    weight = intersection / (union+1e-6) # b*b

                # loss
                loss_ce = self.criterion_ce(logits_w, pids)
                loss_tri = self.criterion_tri(emb_w, pids)

                loss_align = self.align_loss(f_w, f_s, pids, weight)
                loss_uniform = 0.5*(self.uniform_loss(f_w) + self.uniform_loss(f_s))
                loss_domain = 0.5*(self.domain_loss(f_w, self.memory_bank.features, dids) +
                                   self.domain_loss(f_s, self.memory_bank.features, dids))

                with torch.no_grad():
                    self.memory_bank.momentum_update(f_w, pids)

                loss = loss_ce + loss_tri + self.lam * loss_align + loss_uniform + loss_domain

            # update
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            # summing-up
            prec, = accuracy(logits.data, torch.cat([pids, pids]).data)
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_align.update(loss_align.item())
            losses_uniform.update(loss_uniform.item())
            losses_domain.update(loss_domain.item())
            precisions.update(prec[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                    print(f'Epoch: {epoch} [{i + 1}/{iters}] \t'
                          f' Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                          f' L_CE: {losses_ce.val:.3f} ({losses_ce.avg:.3f})  '
                          f' L_Tri: {losses_tri.val:.3f} ({losses_tri.avg:.3f})  '
                          f' L_Align: {losses_align.val:.3f} ({losses_align.avg:.3f})  '
                          f' L_Uniform: {losses_uniform.val:.3f} ({losses_uniform.avg:.3f})  '
                          f' L_Domain: {losses_domain.val:.3f} ({losses_domain.avg:.3f})  '
                          f' Prec: {precisions.val:.2%} ({precisions.avg:.2%})')

    def _parse_data(self, data):
        imgs_w, imgs_s, pids, dids, = data
        return imgs_w.cuda(), imgs_s.cuda(), pids.cuda(), dids.cuda()
    
    def align_loss(self, f_w, f_s, y, w):
        m, n = f_s.size(0), f_w.size(0)
        xx = torch.pow(f_s, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(f_w, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy - 2 * f_s @ f_w.t()

        y = y.unsqueeze(-1) # b*1
        mask = torch.eq(y, y.T).to(f_s.device) # positive pairs
        align_loss = (dist[mask] * w[mask]).sum() / w[mask].sum()
        return align_loss

    def uniform_loss(self, f):
        return torch.pdist(f, p=2).pow(2).mul(-2).exp().mean().log()

    def domain_loss(self, f, c, dids):
        m, n = f.size(0), c.size(0)
        xx = torch.pow(f, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(c, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy - 2 * f @ c.t() # m*n

        domain_mask = torch.eq(dids.unsqueeze(-1), self.memory_bank.labels).float().cuda() # same domain
        sorted_dist, indices = torch.sort(dist + (9999999.) * (1-domain_mask), dim=1, descending=False)
        sorted_dist = sorted_dist[:, 1:m+1] # different class
        return sorted_dist.mul(-2).exp().mean().log()
