class DefaultConfig(object):

    device = 'cuda'                         # 设备
    data_root = './data'                    # 数据存放的根目录
    image_size = 256                        # 图片 resize 大小
    num_workers = 0                         # 进程数
    output_dir = './output'                 # 输出结果
    batch_size = 5

    epoch_max = 100                          # 模型的训练轮次
    epoch_current = 1                       # 当前的训练轮次
    lr = 0.0001                             # 模型的学习率
    model_save_root = './checkpoints'       # 模型的保存目录
    model_path = 'unet_100.pth'                # 已存在的模型

