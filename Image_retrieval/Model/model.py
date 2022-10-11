import torch.nn as nn
import torch


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return torch.mul(x, y)


class Resblock(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(Resblock, self).__init__()
        self.bn = torch.nn.BatchNorm1d(in_channel)
        self.hidden1 = torch.nn.Linear(in_channel, in_channel*2)
        self.hidden2 = torch.nn.Linear(in_channel*2, in_channel)
        self.se = SELayer(in_channel, reduction)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = torch.relu(self.hidden1(x))
        x = self.hidden2(x)
        y = self.se(x)
        return y + residual
    
    
class MlpBlock(nn.Module):

    def __init__(self, hidden_dim=1024, mlp_dim=2048):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dense_1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        shortcut = x 
        x = self.bn(x)
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dense_2(x)
        return x + shortcut
    

class ResAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.res_se = Resblock(self.in_channel)
        self.mlp = MlpBlock(hidden_dim=self.in_channel, mlp_dim=2*self.in_channel)
        
    def forward(self, x):
        x = self.res_se(x)
        x = self.mlp(x)
        return x
    

# SE-ResNet
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden1, num_block, deep_dim, num_class):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1, bias=True)  # hidden layer
        
        self.num_block = num_block
        self.blocks = nn.ModuleList([ResAttention(in_channel=n_hidden1) for _ in range(num_block)])
        self.deep_dim = deep_dim
        
        self.fc = nn.Linear(n_hidden1, self.deep_dim, bias=True)
        self.head = nn.BatchNorm1d(self.deep_dim)
        self.head.bias.requires_grad_(False)
        
        self.classifier = nn.Linear(self.deep_dim, num_class, bias=False)  # output layer
        self.head.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        y = self.hidden1(x)
        
        for l in self.blocks:
            y = l(y)
        
        y = self.fc(y)          # 输出triplet_loss 和center_loss
        feat = self.head(y)                       # 检索的时候余弦相似度
        out = self.classifier(feat)               # 分类
        if self.training:
            return y, out
        else:
            return feat