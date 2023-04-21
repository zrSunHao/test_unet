import torch as t
import numpy as np
from PIL import Image
import os

from configs import DefaultConfig
from models import UNet
from dataprovider import CatSegmentationDataset as Dataset

cfg = DefaultConfig()
# 根据 cuda 的可用情况选择使用 gpu 或 cpu
device = t.device('cpu' if not t.cuda.is_available() else cfg.device)

img_size = (256, 256)
# 加载模型
net = UNet(Dataset.in_channels, Dataset.out_channels)
model_path = cfg.model_save_root + '/' + cfg.model_path
assert os.path.exists(model_path)
state_dict = t.load(model_path)
net.load_state_dict(state_dict)
net.to(device)
net.eval()

# 加载并处理输入图片
ori_img = Image.open('./data/JPEGImages/6.jpg')
img = np.asarray(ori_img.resize(img_size))
img = img/255.                  # 归一化
img = img.transpose(2,0,1)      # 维度转换： hwc -> chw
img = img[np.newaxis,:,:]       # 扩展维度
img = img.astype(np.float32)
img = t.tensor(img).to(device)  # numpy 转为 tensor

# 模型预测
output = net(img)
output = output.cpu().detach().numpy()

# 模型输出转化为 Mask 图片
output = np.squeeze(output)    # 压缩维度
# 根据阈值，判断哪些像素为正例，赋值1，负例，赋值0
output = np.where(output>0.5, 1, 0).astype(np.uint8)

# numpy 数据转为图像
mask = Image.fromarray(output, mode='P') 
# 设置背景和正例的颜色，RGB形式
mask.putpalette([0,0,0, 0,128,0])
mask = mask.resize(ori_img.size)
mask.save(cfg.output_dir + './output.png')

image = ori_img.convert('RGBA')
mask = mask.convert('RGBA')
# 合成
img_mask = Image.blend(image, mask, 0.3)
img_mask.save(cfg.output_dir + './output_mask.png')


