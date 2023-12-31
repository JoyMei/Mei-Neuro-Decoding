import torch
import torch.nn as nn
import torch.nn.functional as F

import sys; sys.path.append("..")



# class SeparableConv2d(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
#         super(SeparableConv2d,self).__init__()

#         self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
#         self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.pointwise(x)
#         return x


# class CPCRNN(nn.Module):
#     def __init__(self, opt):
#         super(CPCRNN, self).__init__()
#         self.conv = nn.Sequential(
#             # SeparableConv2d(in_channels=1, out_channels=32, kernel_size=(1, 64), stride=1, padding=0),
#             # nn.ELU(),
#             nn.Conv2d(in_channels=1, out_channels=8,  kernel_size=(1, 64), stride=1, padding=0),
#             nn.ELU(),
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 16), stride=1, padding=0),
#             nn.ELU(),
#             nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(16, 1), stride=1, padding=0),
#             nn.ELU()
#         )
#         # lstm input: input_size, hidden_size, num_layers
#         self.lstm1 = nn.LSTM(input_size=opt.num_dim, hidden_size=32, num_layers=34)
#         self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=34)
        
#         size = self.get_size()
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(size[1], opt.hidden_size),
#             nn.Dropout(opt.dropout_rate),
#             nn.ReLU(),
#             nn.Linear(opt.hidden_size, opt.num_class)
#         )

#     def get_size(self):
#         x = torch.rand(2, 1, opt.num_channel, opt.num_dim)
#         x1 = self.conv(x)
#         x1 = x1.reshape(x1.size(0), -1)
#         x = x.squeeze(1)
#         x2, (h, c) = self.lstm1(x)
#         x2, _ = self.lstm2(x2, (h, c))
#         x2 = x2.reshape(x2.size(0), -1)
#         out = torch.cat((x1, x2),dim = 1)
#         return out.shape

#     def forward(self, x):
#         x1 = self.conv(x)
#         x1 = x1.reshape(x1.size(0), -1)
#         x = x.squeeze(1)
#         x2, (h, c) = self.lstm1(x)
#         x2, _ = self.lstm2(x2, (h, c))
#         x2 = x2.reshape(x2.size(0), -1)
#         out = torch.cat((x1, x2),dim = 1)
#         self.linear_size = out.shape[1]
#         out = torch.sigmoid(self.fc(out))
#         # print(out.shape)
#         # sys.exit(0)
#         return out
        
class CPCRNN(nn.Module):
    def __init__(self, num_class, input_size):
        super(CPCRNN, self).__init__()
        self.conv = nn.Sequential(
            # SeparableConv2d(in_channels=1, out_channels=32, kernel_size=(1, 64), stride=1, padding=0),
            # nn.ELU(),
            nn.Conv2d(in_channels=1, out_channels=8,  kernel_size=(1, 64), stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 16), stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(16, 1), stride=1, padding=0),
            nn.ELU()
        )
        # lstm input: input_size, hidden_size, num_layers
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=32, num_layers=34)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=34)
        
        size = self.get_size()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size[1], 48),
            nn.Dropout(0.35),
            nn.ReLU(),
            nn.Linear(48,num_class)
        )

    def getsize(self, input_size):
        data = torch.ones(1, 1, input_size[0], input_size[1])
        print(data.shape)
        x1 = self.conv(data)
        print(x1.shape)
        x2, (h, c) = self.lstm1(x)
        x2, _ = self.lstm2(x2, (h, c))
#         out = x.view(x.shape[0], -1)
# #         print(out)tensor([[0.0000, 0.3013, 0.3013,  ..., 0.0000, 0.0000, 0.0650]],
# #        grad_fn=<ViewBackward>)
        out = torch.cat((x1, x2),dim = 1)
        self.linear_size = out.shape[1]
        out = torch.sigmoid(self.fc(out))
        print(out.shape)
        return out.size()

    def forward(self, x):
        x1 = self.conv(x)
        x1 = x1.reshape(x1.size(0), -1)
        x = x.squeeze(1)
        x2, (h, c) = self.lstm1(x)
        x2, _ = self.lstm2(x2, (h, c))
        x2 = x2.reshape(x2.size(0), -1)
        out = torch.cat((x1, x2),dim = 1)
        self.linear_size = out.shape[1]
        out = torch.sigmoid(self.fc(out))
        # print(out.shape)
        # sys.exit(0)
        return out

net = CPCRNN(2,[40,101])
print(net)