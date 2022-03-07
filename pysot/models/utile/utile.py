import torch.nn as nn
import torch.nn.functional as F
import torch as t
import math
from pysot.models.utile.tran import Transformer
class Adadownsamplingnet_tem(nn.Module):

    def __init__(self):
        super(Adadownsamplingnet_tem, self).__init__()
        
        channel=256
        
        self.downsampling1=nn.Sequential(
                nn.Conv2d(256, 256,  kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                )
        self.downsampling2= nn.Sequential(
                nn.Conv2d(128, 128,  kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                )
        
       
        self.conv1= nn.Sequential(
                nn.Conv2d(256, 128,  kernel_size=1, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128,  kernel_size=(1,3), stride=1,padding=(0,1)),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128,  kernel_size=(3,1), stride=1,padding=(1,0)),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(256, 128,  kernel_size=1, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128,  kernel_size=(1,3), stride=1,padding=(0,1)),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128,  kernel_size=(3,1), stride=1,padding=(1,0)),
                ) 
        
        self.conv3 = nn.Sequential(
               nn.ConvTranspose2d(128*2, 256,  kernel_size=1, stride=1),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True),
               )  
        
     
        for modules in [self.conv1,self.conv2,self.conv3,self.downsampling1,self.downsampling2]:
            for l in modules.modules():
               if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0) 
       
    def forward(self, z):
        
        b, c, w, h=z.size()
        
       
        z1=self.downsampling1(z)
        z2=self.conv1(z1)
        
        z3=self.conv2(z)
        z4=self.downsampling2(z3)
       
        z5=self.conv3(t.cat((z2,z4),1))
        
        return z5

class hiftmodule(nn.Module):
    
    def __init__(self,cfg):
        super(hiftmodule, self).__init__()
        
        
        channel=256
        self.conv1=nn.Sequential(
                nn.Conv2d(256, 256,  kernel_size=3, stride=2,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                )
        self.conv2 = nn.Sequential(
               nn.ConvTranspose2d(256*2, 256,  kernel_size=1, stride=1),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True),
               ) 
        self.conv3 = nn.Sequential(
               nn.ConvTranspose2d(256, 256,  kernel_size=2, stride=2),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True),
               nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
               
              
             
                ) 
        
        self.convloc = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=2, stride=2),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),                
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, 4,  kernel_size=3, stride=1,padding=1),
                )
        
        self.convcls = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=2, stride=2),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.row_embed1 = nn.Embedding(50, 256//2)
        self.col_embed1 = nn.Embedding(50, 256//2)
        self.row_embed2 = nn.Embedding(50, 256//2)
        self.col_embed2 = nn.Embedding(50, 256//2)
        self.reset_parameters()
        
        self.trans = Transformer(256, 8,1,1)
        
        self.cls1=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls2=nn.Conv2d(channel, 1,  kernel_size=3, stride=1,padding=1)
        for modules in [self.conv1,self.conv2,self.convloc, self.convcls,self.conv3,
                        self.cls1, self.cls2]:
            for l in modules.modules():
               if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
        
        
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed1.weight)
        nn.init.uniform_(self.col_embed1.weight)
        nn.init.uniform_(self.row_embed2.weight)
        nn.init.uniform_(self.col_embed2.weight)
        
    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def forward(self,x,z,xf,zf):
        
        resx=self.xcorr_depthwise(x, z)
        resd=self.xcorr_depthwise(xf, zf)
        resd=self.conv1(resd)
        res=self.conv2(t.cat((resx,resd),1))
        h1, w1 = 11, 11
        i1 = t.arange(w1).cuda()
        j1 = t.arange(h1).cuda()
        x_emb1 = self.col_embed1(i1)
        y_emb1 = self.row_embed1(j1)

        pos1 = t.cat([
            x_emb1.unsqueeze(0).repeat(h1, 1, 1),
            y_emb1.unsqueeze(1).repeat(1, w1, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(res.shape[0], 1, 1, 1)

        h2, w2 = 22, 22
        i2 = t.arange(w2).cuda()
        j2 = t.arange(h2).cuda()
        x_emb2 = self.col_embed2(i2)
        y_emb2 = self.row_embed2(j2)

        pos2 = t.cat([
            x_emb2.unsqueeze(0).repeat(h2, 1, 1),
            y_emb2.unsqueeze(1).repeat(1, w2, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(res.shape[0], 1, 1, 1)



        b, c, w, h=res.size()
        res1=self.conv3(res)
        res2=self.trans((pos1+res).view(b,256,-1).permute(2, 0, 1),\
                          (pos2+res1).view(b,256,-1).permute(2, 0, 1)) 
                                
                            

       
        res2=res2.permute(1,2,0).view(b,256,22,22)
        loc=self.convloc(res2)
        acls=self.convcls(res2)

        cls1=self.cls1(acls)
        cls2=self.cls2(acls)
        
        return loc,cls1,cls2





