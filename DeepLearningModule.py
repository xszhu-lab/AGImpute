import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np


# -----------------
#  Generate analog values part
# -----------------

class MyDataset(Dataset):
    def __init__(self, data, label, img_size, transform=None):
        self.data = data
        self.label = label
        self.data_cls = np.array(self.label)  #
        self.transform = transform
        self.img_size = img_size
        self.fig_h = self.img_size

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, :].values[0:(self.fig_h * self.fig_h), ].reshape(self.fig_h, self.fig_h, 1).astype(
            'double')  #
        label = np.array(self.data_cls[idx]).astype('int32')  #
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        data = data.transpose((2, 0, 1))

        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(label)
                }


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, max_ncls, channels):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.init_size = img_size // 4
        self.cn1 = 32
        self.l1 = nn.Sequential(nn.Linear(latent_dim, self.cn1 * (self.init_size ** 2)))
        self.l1p = nn.Sequential(nn.Linear(latent_dim, self.cn1 * (img_size ** 2)))

        self.conv_blocks_01p = nn.Sequential(
            nn.BatchNorm2d(self.cn1),
            nn.Conv2d(self.cn1, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1, 0.8),
            nn.ReLU(),
        )

        self.conv_blocks_02p = nn.Sequential(
            nn.Upsample(scale_factor=img_size),
            nn.Conv2d(max_ncls, self.cn1 // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(40, 0.8),
            nn.Conv2d(40, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, noise, label_oh):
        out = self.l1p(noise)
        out = out.view(out.shape[0], self.cn1, self.img_size, self.img_size)
        out01 = self.conv_blocks_01p(out)
        #
        label_oh = label_oh.unsqueeze(2)
        label_oh = label_oh.unsqueeze(2)
        out02 = self.conv_blocks_02p(label_oh)

        out1 = torch.cat((out01, out02), 1)
        out1 = self.conv_blocks_1(out1)
        return out1


class Discriminator(nn.Module):
    def __init__(self, img_size, channels, max_ncls):
        super(Discriminator, self).__init__()

        self.cn1 = 32
        self.down_size0 = 64
        self.down_size = 32
        # pre
        self.pre = nn.Sequential(
            nn.Linear(img_size ** 2, self.down_size0 ** 2),
        )

        self.down = nn.Sequential(
            nn.Conv2d(channels, self.cn1, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.cn1, self.cn1 // 2, 3, 1, 1),
            nn.BatchNorm2d(self.cn1 // 2),
            nn.ReLU(inplace=True),
        )

        self.conv_blocks02p = nn.Sequential(
            nn.Upsample(scale_factor=self.down_size),
            nn.Conv2d(max_ncls, self.cn1 // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(inplace=True),
        )

        down_dim = 24 * (self.down_size) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(16, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(24, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, channels, 2, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img, label_oh):
        out00 = self.pre(img.view((img.size()[0], -1))).view((img.size()[0], 1, self.down_size0, self.down_size0))
        out01 = self.down(out00)

        label_oh = label_oh.unsqueeze(2)
        label_oh = label_oh.unsqueeze(2)
        out02 = self.conv_blocks02p(label_oh)
        ####
        out1 = torch.cat((out01, out02), 1)
        ######
        out = self.fc(out1.view(out1.size(0), -1))
        out = self.up(out.view(out.size(0), 24, self.down_size, self.down_size))
        return out


class AutoEncoder(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_shape, 64),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(128, 128),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(128, 128),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(64, out_dim))
        self.decoder = nn.Sequential(nn.Linear(out_dim, 64),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(128, 128),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(128, 128),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(64, input_shape))

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


def one_hot(batch, depth):
    ones = torch.eye(depth)
    return ones.index_select(0, batch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
