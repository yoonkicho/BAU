from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import gc
import collections

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from bau import datasets
from bau import models
from bau.trainers import BAUTrainer
from bau.models.memory import MemoryBank
from bau.evaluators import Evaluator, extract_features
from bau.utils.data import IterLoader, MultiSourceTrainDataset
from bau.utils.data import transforms as T
from bau.utils.data.randaugment import RandAugment
from bau.utils.data.sampler import RandomIdentitySampler
from bau.utils.data.preprocessor import Preprocessor, TwoViewPreprocessor
from bau.utils.logging import Logger
from bau.utils.lr_scheduler import WarmupMultiStepLR

best_mAP = 0


def get_data(name, data_dir):
    root = data_dir
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers, num_instances, images_dir):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    
    transforms_w = T.Compose([T.Resize((height, width), interpolation=3),
                              T.RandomHorizontalFlip(p=0.5),
                              T.Pad(10),
                              T.RandomCrop((height, width)),
                              T.ToTensor(),
                              normalizer,])

    transforms_s = T.Compose([T.Resize((height, width), interpolation=3),
                              T.RandomHorizontalFlip(p=0.5),
                              T.Pad(10),
                              T.RandomCrop((height, width)),
                              T.RandomApply([T.ColorJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0), saturation=(0.5, 2.0), hue=(-0.1, 0.1))], p=args.prob),
                              T.ToTensor(),
                              normalizer,
                              T.RandomErasing(mean=[0.485, 0.456, 0.406], probability=args.prob),])
    
    transforms_s.transforms.insert(0, T.RandomApply([RandAugment()], p=args.prob))

    train_set = sorted(dataset.train)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomIdentitySampler(train_set, batch_size, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(DataLoader(TwoViewPreprocessor(train_set, root=images_dir, transform_w=transforms_w, transform_s=transforms_s),
                                         batch_size=batch_size, num_workers=workers, sampler=sampler,shuffle=not rmgs_flag, pin_memory=True, 
                                         drop_last=True), length=None)
    return train_loader


def get_memory_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])
    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(TwoViewPreprocessor(testset, root=dataset.images_dir, transform=test_transformer),
                             batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)
    return test_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])
    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))
    test_loader = DataLoader(Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
                             batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)
    return test_loader


def create_model(args, num_classes):
    model = models.create(args.arch, num_classes=num_classes)
    model.cuda()
    model = torch.nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global best_mAP
    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # organize dataset
    source_dataset = []
    for ds in args.source_dataset:
        dataset = get_data(ds, args.data_dir)
        source_dataset.append(dataset)
        
    train_dataset = MultiSourceTrainDataset(datasets=source_dataset)
    train_loader = get_train_loader(args, train_dataset, args.height, args.width, args.batch_size,
                                    args.workers, args.num_instances, train_dataset.images_dir)
    
    memory_loader = get_memory_loader(train_dataset, args.height, args.width, args.batch_size, args.workers,
                                        testset=sorted(train_dataset.train))
    
    test_dataset = get_data(args.target_dataset, args.data_dir)
    test_loader = get_test_loader(test_dataset, args.height, args.width, args.batch_size, args.workers)

    # backbone
    num_classes = train_dataset.num_train_pids
    model = create_model(args, num_classes=num_classes)

    # memory bank with init
    memory_bank = MemoryBank(2048, num_classes).cuda()

    features, _ = extract_features(model, memory_loader, print_freq=50)
    features_dict = collections.defaultdict(list)
    for f, pid, _, _ in sorted(train_dataset.train):
        features_dict[pid].append(features[f].unsqueeze(0))
    centroids = [torch.cat(features_dict[pid],0).mean(0) for pid in sorted(features_dict.keys())]
    centroids = torch.stack(centroids, 0)
    centroids = torch.nn.functional.normalize(centroids, dim=1)
    memory_bank.features = centroids.cuda()

    domain_offset = 0
    domain_labels = []
    for ds in source_dataset:
        domain_labels.append(torch.ones(ds.num_train_pids, dtype=torch.long)*domain_offset)
        domain_offset += 1
    domain_labels = torch.cat(domain_labels)
    memory_bank.labels = domain_labels.cuda()

    del source_dataset, memory_loader, features_dict, centroids, domain_labels
    gc.collect()

    # evaluator
    evaluator = Evaluator(model)

    # optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # train
    trainer = BAUTrainer(model, memory_bank, num_classes, args.margin, args.lam, args.k)

    for epoch in range(args.epochs):
        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, iters=args.iters)
        lr_scheduler.step()

        if (epoch+1 in args.eval_epochs) or (epoch == args.epochs-1):
            print('\n* Finished epoch {:3d}'.format(epoch))
            mAP = evaluator.evaluate(test_loader, test_dataset.query, test_dataset.gallery, cmc_flag=False)
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), osp.join(args.logs_dir, 'best.pth'))
            print('current mAP: {:5.1%}  best mAP: {:5.1%}'.format(mAP, best_mAP))

    torch.save(model.state_dict(), osp.join(args.logs_dir, 'last.pth'))

    # results
    model.load_state_dict(torch.load(osp.join(args.logs_dir, 'best.pth')))
    evaluator.evaluate(test_loader, test_dataset.query, test_dataset.gallery, cmc_flag=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Balancing Alignment and Uniformity (BAU)")
    
    # data
    parser.add_argument('-ds', '--source-dataset', nargs='+', type=str, default=['msmt17', 'market1501', 'cuhksysu'])
    parser.add_argument('-dt', '--target-dataset', type=str, default='cuhk03')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-n', '--num-instances', type=int, default=4,
                        help='each minibatch consist of '
                             '(batch_size // num_instances) identities, and '
                             'each identity has num_instances instances, '
                             'default: 0 (NOT USE)')
    parser.add_argument('--height', type=int, default=256, help='input height')
    parser.add_argument('--width', type=int, default=128, help='input width')

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs/test'))

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-epochs', nargs='+', type=int, default=[50,55])

    # train
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--margin', type=float, default=0.3, help='margin parameter for triplet loss')
    parser.add_argument('--lam', type=float, default=1.5, help='weighting parameter for alignment loss')
    parser.add_argument('--k', type=int, default=10, help='k-NN parameter for weighting strategy')
    parser.add_argument('--prob', type=float, default=0.5, help='probability of applying data augmentation to inputs')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=500, help='iteration for each epcoh')
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 50], help='milestones for the learning rate decay')
    main()