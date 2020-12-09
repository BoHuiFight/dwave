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
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),dilation=step)

    def forward(self, X):

        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) * torch.sigmoid(self.conv2(X))
        rX=torch.flip(X, [3])
        rtemp=self.conv1(rX) * torch.sigmoid(self.conv2(rX))
        temp=torch.cat((temp, rtemp), 3)
        out = temp.permute(0, 2, 3, 1)

        return out

class PCNLayer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=2,step=2):
        super(PCNLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),dilation=step)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),dilation=step)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),dilation=step)
    def forward(self, x,s2rDic,trajDic):

        #print(x.shape)
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
            rtemp1=torch.flip(temp1, [3])
            rtemp2=self.conv1(rtemp1) * torch.sigmoid(self.conv2(rtemp1))
            temp2=torch.cat((temp2, rtemp2), 3)
            temp.append(temp2)            
          temp=torch.stack(temp)
          temp=temp.mean(0)
          h.append(temp)
        h=torch.stack(h)
        #print(h.shape)
        outputs = h.permute(1,0,2,3,4)
        #print(outputs.shape)
        #outputs=torch.stack(outputs)
        return outputs


class PCNBlock(nn.Module):


    def __init__(self, in_channels, out_channels,num_nodes):
        super(PCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.PCNLayer = PCNLayer(in_channels=out_channels,
                                   out_channels=out_channels)

    def forward(self, X,s2rDic,trajDic):

        x = self.temporal1(X)
        x=x.permute(1, 0, 2, 3)
        output=self.PCNLayer(x,s2rDic,trajDic)
        return output


class PCN(nn.Module):

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):

        super(PCN, self).__init__()
        self.block1 = PCNBlock(in_channels=num_features,
                               out_channels=16,num_nodes=num_nodes)
        self.fully = nn.Linear((num_timesteps_input - 2 * 2)*2*3*2 * 16,
                               num_timesteps_output,bias = True)


    def forward(self,X,r2sDic,s2rDic,trajDic):

        temp=[]
        for seg in s2rDic:
          temp.append(X[:,s2rDic[seg],:,:])
        X=torch.stack(temp)
        X=X.permute((1,0,2,3))
        out1 = self.block1(X,s2rDic,trajDic)
        out3=out1
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        #print(out4.shape)
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


