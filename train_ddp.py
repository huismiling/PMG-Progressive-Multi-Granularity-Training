from __future__ import print_function
from copy import deepcopy
import os
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from utils import *

import argparse

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None, args=None):
    init_seeds(args.seed)
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    dist.init_process_group(backend="nccl")
    # rank = dist.get_rank()
    device_id = RANK  # % torch.mlu.device_count()
    print(f"device_id:{device_id}")


    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        print(RANK, worker_id, worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    # with torch_distributed_zero_first(rank):
    trainset = torchvision.datasets.ImageFolder(root='CUB_200_2011/dataset/train/', transform=transform_train)
    sampler = None if RANK == -1 else torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                batch_size=batch_size, 
                                shuffle=True and sampler is None, 
                                sampler=sampler, 
                                worker_init_fn=seed_worker, generator=g,
                                num_workers=4)

    # Model
    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)

    # GPU
    device = torch.device("cuda", device_id)
	
    net = net.to(device)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    netp = DDP(net, device_ids=[device_id], find_unused_parameters=True)

    # cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for epoch in range(start_epoch, nb_epoch):
        if RANK != -1:
            trainloader.sampler.set_epoch(epoch)
            dist.barrier()

        print(f'\nrank: {RANK} Epoch: {epoch}')
        netp.train()

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            # if inputs.shape[0] < batch_size:
            #     continue
            if use_cuda:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # Step 1
            optimizer.zero_grad()
            inputs1 = jigsaw_generator(inputs, 8)
            # output_1, _, _, _ = netp(inputs1)
            output_1 = netp(inputs1, stage=1)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            # Step 2
            optimizer.zero_grad()
            inputs2 = jigsaw_generator(inputs, 4)
            # _, output_2, _, _ = netp(inputs2)
            output_2 = netp(inputs2, stage=2)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # Step 3
            optimizer.zero_grad()
            inputs3 = jigsaw_generator(inputs, 2)
            # _, _, output_3, _ = netp(inputs3)
            output_3 = netp(inputs3, stage=3)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()

            # Step 4
            optimizer.zero_grad()
            # _, _, _, output_concat = netp(inputs)
            output_concat = netp(inputs, stage=4)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()

            #  training log
            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx % 50 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))

        if RANK in {-1, 0}:
            train_acc = 100. * float(correct) / total
            train_loss = train_loss / (idx + 1)
            with open(exp_dir + '/results_train_multi.txt', 'a') as file:
                file.write(
                    'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                    epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                    train_loss4 / (idx + 1)))

            if epoch < 5 or epoch >= 80:
                print("\nstart test...")
                snet = deepcopy(netp.module)
                val_acc, val_acc_com, val_loss = test(snet, CELoss, 3, device=device)
                if val_acc_com > max_val_acc:
                    max_val_acc = val_acc_com
                    snet = deepcopy(netp.module)
                    torch.save(snet, './' + store_name + '/model.pth')
                with open(exp_dir + '/results_test_multi.txt', 'a') as file:
                    file.write('Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                    epoch, val_acc, val_acc_com, val_loss))
            else:
                snet = deepcopy(netp.module)
                torch.save(snet, './' + store_name + '/model.pth')

if __name__=="__main__":

    args = parse_opt()
    train(nb_epoch=200,        # number of epoch
        batch_size=16,         # batch size
        store_name='bird',     # folder for output
        resume=False,          # resume training from checkpoint
        start_epoch=0,         # the start epoch number when you resume the training
        model_path='',
        args=args)         # the saved model where you want to resume the training
