import torch as t
import torch.nn as nn

from .block import Block

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.conv_encoder_1 = Block(in_channels, features)
        self.conv_encoder_2 = Block(features, features*2)
        self.conv_encoder_3 = Block(features*2, features*4)
        self.conv_encoder_4 = Block(features*4, features*8)

        self.bottleneck = Block(features*8, features*16)

        self.upconv4 = nn.ConvTranspose2d(in_channels=features*16,
                                          out_channels=features*8,
                                          kernel_size=2,
                                          stride=2)
        self.conv_decoder_4 = Block(features*8 *2, 
                                    features*8)
        self.upconv3 = nn.ConvTranspose2d(in_channels=features*8,
                                          out_channels=features*4,
                                          kernel_size=2,
                                          stride=2)
        self.conv_decoder_3 = Block(features*4 *2, 
                                    features*4)
        self.upconv2 = nn.ConvTranspose2d(in_channels=features*4,
                                          out_channels=features*2,
                                          kernel_size=2,
                                          stride=2)
        self.conv_decoder_2 = Block(features*2 *2, 
                                    features*2)
        self.upconv1 = nn.ConvTranspose2d(in_channels=features*2,
                                          out_channels=features,
                                          kernel_size=2,
                                          stride=2)
        self.conv_decoder_1 = Block(features *2, 
                                    features)

        self.conv = nn.Conv2d(in_channels=features,
                              out_channels=out_channels,
                              kernel_size=1)


    def forward(self, x):
        # print('输入', x.size())
        conv_encoder_1_1 = self.conv_encoder_1(x)
        # print('conv_encoder_1_1 输出', conv_encoder_1_1.size())
        conv_encoder_1_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_1_1)
        # print('conv_encoder_1_2 输出', conv_encoder_1_2.size())

        conv_encoder_2_1 = self.conv_encoder_2(conv_encoder_1_2)
        # print('conv_encoder_2_1 输出', conv_encoder_2_1.size()) 
        conv_encoder_2_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_2_1)
        # print('conv_encoder_2_2 输出', conv_encoder_2_2.size())

        conv_encoder_3_1 = self.conv_encoder_3(conv_encoder_2_2)
        # print('conv_encoder_3_1 输出', conv_encoder_3_1.size())
        conv_encoder_3_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_3_1)
        # print('conv_encoder_3_2 输出', conv_encoder_3_2.size())

        conv_encoder_4_1 = self.conv_encoder_4(conv_encoder_3_2)
        # print('conv_encoder_4_1 输出', conv_encoder_4_1.size())
        conv_encoder_4_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_4_1)
        # print('conv_encoder_4_2 输出', conv_encoder_4_2.size())

        bottleneck = self.bottleneck(conv_encoder_4_2)
        # print('bottleneck 输出', bottleneck.size())

        conv_decoder_4_1 = self.upconv4(bottleneck)
        # print('conv_decoder_4_1 输出', conv_decoder_4_1.size())
        conv_decoder_4_2 = t.cat((conv_decoder_4_1, conv_encoder_4_1), dim=1)
        # print('conv_decoder_4_2 输出', conv_decoder_4_2.size())
        conv_decoder_4_3 = self.conv_decoder_4(conv_decoder_4_2)
        # print('conv_decoder_4_3 输出', conv_decoder_4_3.size())

        conv_decoder_3_1 = self.upconv3(conv_decoder_4_3)
        # print('conv_decoder_3_1 输出', conv_decoder_3_1.size())
        conv_decoder_3_2 = t.cat((conv_decoder_3_1, conv_encoder_3_1), dim=1)
        # print('conv_decoder_3_2 输出', conv_decoder_3_2.size())
        conv_decoder_3_3 = self.conv_decoder_3(conv_decoder_3_2)
        # print('conv_decoder_3_3 输出', conv_decoder_3_3.size())

        conv_decoder_2_1 = self.upconv2(conv_decoder_3_3)
        # print('conv_decoder_2_1 输出', conv_decoder_2_1.size())
        conv_decoder_2_2 = t.cat((conv_decoder_2_1, conv_encoder_2_1), dim=1)
        # print('conv_decoder_2_2 输出', conv_decoder_2_2.size())
        conv_decoder_2_3 = self.conv_decoder_2(conv_decoder_2_2)
        # print('conv_decoder_2_3 输出', conv_decoder_2_3.size())

        conv_decoder_1_1 = self.upconv1(conv_decoder_2_3)
        # print('conv_decoder_1_1 输出', conv_decoder_1_1.size())
        conv_decoder_1_2 = t.cat((conv_decoder_1_1, conv_encoder_1_1), dim=1)
        # print('conv_decoder_1_2 输出', conv_decoder_1_2.size())
        conv_decoder_1_3 = self.conv_decoder_1(conv_decoder_1_2)
        # print('conv_decoder_1_3 输出', conv_decoder_1_3.size())

        out = self.conv(conv_decoder_1_3)
        # print('conv 输出', out.size(), '\n')
        out = t.sigmoid(out)
        return out
    
'''
# 256 * 256 的图片

输入                    [2, 3,  256, 256]

conv_encoder_1_1 输出   [2, 32, 256, 256]
conv_encoder_1_2 输出   [2, 32, 128, 128]

conv_encoder_2_1 输出   [2, 64, 128, 128]
conv_encoder_2_2 输出   [2, 64, 64,   64]

conv_encoder_3_1 输出   [2, 128, 64,  64]
conv_encoder_3_2 输出   [2, 128, 32,  32]

conv_encoder_4_1 输出   [2, 256, 32,  32]
conv_encoder_4_2 输出   [2, 256, 16,  16]

bottleneck 输出         [2, 512, 16,  16]

conv_decoder_4_1 输出   [2, 256, 32,  32]
conv_decoder_4_2 输出   [2, 512, 32,  32]
conv_decoder_4_3 输出   [2, 256, 32,  32]

conv_decoder_3_1 输出   [2, 128, 64,  64]
conv_decoder_3_2 输出   [2, 256, 64,  64]
conv_decoder_3_3 输出   [2, 128, 64,  64]

conv_decoder_2_1 输出   [2, 64, 128, 128]
conv_decoder_2_2 输出   [2, 128,128, 128]
conv_decoder_2_3 输出   [2, 64, 128, 128]

conv_decoder_1_1 输出   [2, 32, 256, 256]
conv_decoder_1_2 输出   [2, 64, 256, 256]
conv_decoder_1_3 输出   [2, 32, 256, 256]

conv 输出               [2, 1,  256, 256]
'''