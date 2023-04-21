import os
import numpy as np
import torch as t

from torch.utils.data import Dataset
from PIL import Image

class  CatSegmentationDataset(Dataset):
    
    in_channels=3   # 模型输入是 3 通道
    out_channels=1  # 模型输出是 1 通道

    def __init__(self, image_dir, image_size=32):
        super(CatSegmentationDataset, self).__init__()

        self.image_size = image_size
        print("Reading image...")

        # 原图所在位置
        img_root_path = image_dir  + '/JPEGImages/'
        # Mask 所在位置
        mask_root_path = image_dir + '/SegmentationClassPNG/'
        # 将图片与 Mask 地址，分别存在 img_slices 与 mask_slices 中
        self.img_slices = []
        self.mask_slices = []
        for img_name in os.listdir(img_root_path):
            mask_name = img_name.split('.')[0] + '.png'

            img_path = img_root_path + img_name
            self.img_slices.append(img_path)
            mask_path = mask_root_path + mask_name
            self.mask_slices.append(mask_path)

    def __len__(self):
        return len(self.img_slices)
    
    def __getitem__(self, idx):
        resize_w_h = (self.image_size, self.image_size)

        img_path = self.img_slices[idx]
        img_file = Image.open(img_path).resize(resize_w_h)
        img = np.asarray(img_file) / 255.
        img = img.transpose(2, 0, 1) # hwc（numpy 顺序） =>chw（tensor 顺序）
        img = img.astype(np.float32)
        img = t.tensor(img)

        mask_path = self.mask_slices[idx]
        mask_file = Image.open(mask_path).resize(resize_w_h)
        mask = np.asarray(mask_file)
        mask = mask[np.newaxis, :, :]  # mask 是单通道数据，需加一个维度
        mask = mask.astype(np.float32)
        mask = t.tensor(mask)

        return img, mask

