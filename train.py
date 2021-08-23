import os
import sys
import cv2
import test
import yaml
import torch
import random
import logging
import warnings
import argparse
import math, time
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from torch.cuda import amp
from utils.config import Config 
from model.model import Detector
from utils.dist import is_parallel
from dataset.dataset import Transforms
from dataset.VOC_dataset import VOCDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import set_logging, seeds, select_device, DDP_launch, plot_images, one_cycle
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

mixed_precision = True
# Method 2: attention ZeroDivisionError: float division by zero
# try:  # Mixed precision training https://github.com/NVIDIA/apex [not recommand]
#     from apex import amp
# except:
#     print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
#     mixed_precision = False  # not installed

def train(opt,hyp,device):
    seeds(opt.seed)
    rank, weights  = opt.local_rank, opt.weights
    epochs = Config.epochs
    cuda = device.type != 'cpu'

    model = Detector().to(device) # if torch.distributed.launch than move model to all GPU
    if hyp['Adam']:
        optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr'], momentum=hyp['momentum'], nesterov=True)
    
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.scheduler == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
    elif opt.scheduler == "linear_lr":
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)            # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = ckpt['model']
        # optimizer = ckpt['optimizer']

    # Method2: Mixed precision training https://github.com/NVIDIA/apex [not recommand]
    # if mixed_precision:
    #     logger.info("amp")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
    # Attention: ZeroDivisionError: float division by zero

    # DDP mode
    if cuda and rank != -1: # DP and CPU has no syncbn
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=False)
    
    # DP mode: move model to all GPU
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train_dataset = VOCDataset(root_dir=Config.train_path, resize_size=[800,1333],
                            split=Config.train_split, use_difficult=False, is_train=True, augment = Transforms())
    #Cpu and DP set None
    workers = min([os.cpu_count() // opt.world_size, opt.batch_size if opt.batch_size > 1 else 0, opt.workers])  # number of workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None 
    train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = opt.batch_size, num_workers = workers,
                                                    shuffle = False, sampler = train_sampler, pin_memory = True,
                                                    collate_fn = train_dataset.collate_fn, worker_init_fn=np.random.seed(0))
    scaler = amp.GradScaler(enabled=cuda)
    batch_nums = len(train_dataloader)                                   # number of batches
    warmup_thresh = max(round(hyp['warmup_epochs'] * batch_nums), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    start_epoch, best_map = 0, 0
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(4, device=device)        # mean losses
        if rank != -1:
            train_dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(train_dataloader)
        logger.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'cls', 'reg', 'cnt', 'total', 'lr'))
        
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total = batch_nums)    # progress bar
        optimizer.zero_grad()
        for i, (imgs, bobx_targets, classes_targets, paths, index)  in pbar:
            iteration_nums = i + batch_nums * epoch  # number integrated batches (since train start)
            imgs, bobx_targets, classes_targets = imgs.to(device),bobx_targets.to(device), classes_targets.to(device)

            # Warmup
            xi = [0, warmup_thresh-1]  # x interp
            for param in optimizer.param_groups:
                param['lr'] = np.interp(iteration_nums, xi, [ 0.0 ,hyp['lr']])
                if 'momentum' in param:
                    param['momentum'] = np.interp(iteration_nums, xi, [hyp['warmup_momentum'], hyp['momentum']])
            
            # Forward
            loss, loss_items = model(imgs, bobx_targets, classes_targets) # loss scaled by batch_size
            if rank != -1:
                loss *= opt.world_size                                    # gradient averaged between devices in DDP mode??

            # Backward
            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                # Method 2
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
            else:                       # DP (1,n) n=batch_size_1GPU
                loss.mean().backward()  # DP Mode RuntimeError: grad can be implicitly created only for scalar outputs
                optimizer.step()
            
            optimizer.zero_grad()
            
            if rank in [-1, 0]:
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                if opt.world_size == 1:
                    s = ('%10s' * 2 + '%10.4g' * 5) % (                    # update mean losses in 1pic
                        '%g/%g' % (epoch, epochs - 1), mem, loss_items[0].mean(), loss_items[1].mean(), loss_items[2].mean(), loss_items[3].mean() ,optimizer.param_groups[0]['lr'])
                else:
                    mloss = (mloss * i + torch.cat(loss_items)) / (i + 1)  # update mean losses in 1epoch
                    s = ('%10s' * 2 + '%10.4g' * 5) % (
                        '%g/%g' % (epoch, epochs - 1), mem, *mloss ,optimizer.param_groups[0]['lr'])
                pbar.set_description(s)

                # Plot
                # if ni < 3:
                #     f = Config.save_dir / f'train_batch{ni}.jpg'  # filename
                #     # Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                #     result = plot_images(images=imgs, targets=(bobx_targets,classes_targets), paths=paths, fname=f)

                # elif plots and ni == 10:
                #     wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                #                                   save_dir.glob('train*.jpg') if x.exists()]})
    
        # Scheduler
        scheduler.step()
        
        ckpt = {'epoch': epoch,
                # 'model': deepcopy(model.module if is_parallel(model) else model).half(),
                'model': deepcopy(model.module if is_parallel(model) else model),
                'optimizer': optimizer.state_dict()}
        torch.save(ckpt, Config.last)
        del ckpt
        
        if rank in [-1, 0]:
            test_dataset = VOCDataset(root_dir = Config.test_path, resize_size = [800,1333],
                                split=Config.test_split, use_difficult = False, is_train = True, augment = None)
            test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = Config.test_batch, 
                                                        num_workers  = workers, shuffle = False, pin_memory = True, 
                                                        collate_fn = test_dataset.collate_fn, worker_init_fn=np.random.seed(0))
            # map50, map5095 = test.test(model, test_dataloader, distributed=False)
            
            # Method2
            ap, map50 = test.test(model, test_dataloader, distributed=False)
            for key,value in ap.items():
                    logger.info(f'ap for {test_dataset.id2name[int(key)]}:\t\t{value}')
            if epoch == epochs:
                for key,value in ap.items():
                    logger.info(f'ap for {test_dataset.id2name[int(key)]}:\t\t{value}')
            logger.info("mAP@0.5  =====>  %.3f\n"%map50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument("--seed", type=int, default=100, help="random seed")
    parser.add_argument('--hyp', type=str, default='utils/voc_hyp.yaml', help='hyperparameters path')
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, recive from torch.distributed.launch module in terminal')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu, defaul for DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--scheduler', default='StepLR', help='linear LR')

    opt = parser.parse_args()
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1 # nproc_per_node*node(nn.DataParallel 需要所有的GPU都在一个节点（一台机器）上，且并不支持 Apex 的混合精度训练)
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    opt.total_batch_size = opt.batch_size   #for cpu/1GPU/4GPU1process(DPmode)

    # CPU or 1GPU or DPmode
    set_logging(opt.local_rank) # set_logging(opt.global_rank)
    logger.info(opt)

    if opt.world_size == 1:
        mixed_precision = False
    device = select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size) #0GPU or CPU first load optional paras

    # DDP mode
    if opt.local_rank != -1:
        device = DDP_launch(opt)
    
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    logger.info(hyp)
    train(opt, hyp, device)