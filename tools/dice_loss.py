import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_label):
        assert y_pred.size() == y_label.size()

        # 取预测结果的第一个通道，并将维度转换为 1 维
        # 256*256的图片 [2, 1, 32, 32] --> [2048]
        y_pred = y_pred[:, 0].contiguous().view(-1)
        # 256*256的图片 [2, 1, 32, 32] --> [2048]
        y_label = y_label[:, 0].contiguous().view(-1)

        intersection = (y_pred * y_label).sum()
        combine = y_pred.sum() + y_label.sum()
        dsc = (2 * intersection + self.smooth) / (combine + self.smooth)

        return 1 - dsc
        