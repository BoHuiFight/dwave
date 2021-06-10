import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,step=2):

        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),dilation=step)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),dilation=step)
    def forward(self, X):

        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) * torch.sigmoid(self.conv2(X))
        out = temp.permute(0, 2, 3, 1)

        return out

class PCNLayer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=2,step=2):
        super(PCNLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),dilation=step)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),dilation=step)
    def forward(self, x,r2sDic,s2rDic,trajDic):

        keys=[*s2rDic]
        h=[]
        for s in s2rDic:
          temp=[]
          for traj in trajDic[s]:
            temp1=[]
            for t in traj:
              temp1.append(x[t,:,:,:])
            temp1=torch.stack(temp1)
            temp1=temp1.permute(1, 3, 2, 0)
            temp2 = self.conv1(temp1) * torch.sigmoid(self.conv2(temp1))
            temp.append(temp2)            
          temp=torch.stack(temp)
          temp=temp.mean(0)
          h.append(temp)
        h=torch.stack(h)
        outputs = h.permute(1,0,2,3,4)
        return outputs


class PCNBlock(nn.Module):


    def __init__(self, in_channels, out_channels,num_nodes):
        super(PCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.PCNLayer = PCNLayer(in_channels=out_channels,
                                   out_channels=out_channels)

    def forward(self, X,X_daily,X_weekly,X_coarse,r2sDic,s2rDic,trajDic):

        x = self.temporal1(X)
        x_daily = self.temporal1(X_daily)
        x_weekly = self.temporal1(X_weekly)
        x=torch.cat((x, x_daily,x_weekly), 2)
        x=x.permute(1, 0, 2, 3)
        output=self.PCNLayer(x,r2sDic,s2rDic,trajDic)
        return output


class PCN(nn.Module):

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):

        super(PCN, self).__init__()
        self.block1 = PCNBlock(in_channels=num_features,
                               out_channels=16,num_nodes=num_nodes)
        self.fully = nn.Linear((num_timesteps_input - 2 * 2+2+8)*3 * 16,
                               num_timesteps_output,bias = True)


    def forward(self,X,X_daily,X_weekly,X_coarse,r2sDic,s2rDic,trajDic):

        temp=[]
        tempD=[]
        tempW=[]
        tempC=[]
        for seg in s2rDic:
          temp.append(X[:,s2rDic[seg],:,:])
          tempD.append(X_daily[:,s2rDic[seg],:,:])
          tempW.append(X_weekly[:,s2rDic[seg],:,:])
          tempC.append(X_coarse[:,s2rDic[seg],:,:])

        X=torch.stack(temp)
        X_daily=torch.stack(tempD)
        X_weekly=torch.stack(tempW)
        X_coarse=torch.stack(tempC)
        X=X.permute((1,0,2,3))
        X_daily=X_daily.permute((1,0,2,3))
        X_weekly=X_weekly.permute((1,0,2,3))
        X_coarse=X_coarse.permute((1,0,2,3))
        out1 = self.block1(X,X_daily,X_weekly,X_coarse,r2sDic,s2rDic,trajDic)
        out3=out1
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        temp2=[]
        for r in r2sDic:
          temp=[]
          for s in r2sDic[r]:
            temp.append(out4[:,[*s2rDic].index(s),:])
          temp_new=torch.stack(temp)
          temp_new=torch.mean(temp_new,dim=0)
          temp2.append(temp_new)
        out4=torch.stack(temp2).permute((1,0,2))
        return out4

