import torch as t
from torch.utils.data import DataLoader
from torch import optim
import tqdm
import numpy as np
import os

from configs import DefaultConfig
from models import UNet
from tools import DiceLoss
from dataprovider import CatSegmentationDataset as Dataset

cfg = DefaultConfig()
# 根据 cuda 的可用情况选择使用 gpu 或 cpu
device = t.device('cpu' if not t.cuda.is_available() else cfg.device)

# 加载训练数据集
dataset_train = Dataset(image_dir=cfg.data_root, image_size=cfg.image_size)
loader_train = DataLoader(dataset=dataset_train,
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          num_workers=0)

# 实例化 UNet 网络模型
net = UNet(Dataset.in_channels, Dataset.out_channels)
model_path = cfg.model_save_root + cfg.model_path
if os.path.exists(model_path):
    state_dict = t.load(model_path)
    net.load_state_dict(state_dict)
net.to(device)

# 损失函数
dsc_loss = DiceLoss()
# 优化方法
optimizer = optim.Adam(net.parameters(), lr = cfg.lr)

# 训练 n 个Epoch
for epoch in range(cfg.epoch_max):
    if((epoch+1)<cfg.epoch_current):
        continue
    net.train()
    loss_list = []
    for i, data in enumerate(loader_train):
        x, y_lable = data
        x, y_lable = x.to(device), y_lable.to(device)

        optimizer.zero_grad()
        y_pred = net(x)
        loss = dsc_loss(y_pred, y_lable)
        loss_list.append(loss.item())
        loss.backward()
    t.save(net.state_dict(), cfg.model_save_root + '/unet_{}.pth'.format(epoch+1))
    print('epoch ', epoch+1, 'Loss', np.mean(loss_list)) 
