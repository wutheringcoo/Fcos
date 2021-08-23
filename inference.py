import torch
import torch.nn as nn
import numpy as np

from dataset.VOC_dataset import VOCDataset
from utils.utils import Config, seeds
from test import test

seeds(1000)

model =  torch.load('/data-nbd/wuxc/111111/last.pt',map_location='cuda:0')['model']

test_dataset = VOCDataset(root_dir = Config.test_path, resize_size = [800,1333],
                                split=Config.test_split, use_difficult = False, is_train = True, augment = None)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = Config.test_batch, 
                                            num_workers  = 8, shuffle = False, pin_memory = False, 
                                            collate_fn = test_dataset.collate_fn, worker_init_fn=np.random.seed(0))
map50, map5095 = test(model, test_dataloader, distributed=False)


