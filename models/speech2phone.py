"""
SpiraConv models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generic_utils import Mish

class Speech2PhoneV2(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config):
        super(Speech2PhoneV2, self).__init__()
        self.config = config
        self.audio = self.config['audio']
        self.padding_with_max_lenght = self.config.dataset['padding_with_max_lenght'] or self.config.dataset['split_wav_using_overlapping']
        self.max_seq_len = self.config.dataset['max_seq_len']
        print("Model Inicialized With Feature %s "%self.config.audio['feature'])
        if self.config.audio['feature'] == 'spectrogram':
            self.num_feature = self.config.audio['num_freq']
        elif self.config.audio['feature'] == 'melspectrogram':
            self.num_feature = self.config.audio['num_mels']
        elif self.config.audio['feature'] == 'mfcc':
            self.num_feature = self.config.audio['num_mfcc']
        else:
            self.num_feature = None
            raise ValueError('Feature %s Dont is supported'%self.config.audio['feature'])

        self.dropout = nn.Dropout(p=0.83)
        self.fc0 = nn.Linear(self.max_seq_len*self.num_feature, 40)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(40*2, (int(self.max_seq_len/self.config.dataset['window_len'])+1)*self.num_feature)

    def forward(self, x):
        x = self.torchfb(x)+1e-6
        # print(x.shape)
        #x = self.instancenorm(x.log()).detach().transpose(1, 2)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc0(x)
        # compute CReLU
        x = torch.cat((self.relu(x), self.relu(-x)), 1)
        print(x.shape)
        x = x.view(x.size()[0], -1)
        #x = self.encoder(x)
        # print(x.shape)
        # print(x.shape)
        x = self.fc(x)
        return x
