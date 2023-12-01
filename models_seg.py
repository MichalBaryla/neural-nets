import torch
import torch.nn as nn
import torch.nn.functional as F

def block_down(inc,outc,mid_big=False,kern_size_desc=True):
    
    mid = 0
    if mid_big: mid=outc
    else: mid=inc

    if kern_size_desc:
        c1 = nn.Conv2d(inc,mid,4,2,1)
        c2 = nn.Conv2d(mid,outc,3,1,1)
        
    else:
        c1 = nn.Conv2d(inc,mid,3,1,1)
        c2 = nn.Conv2d(mid,outc,4,2,1)
        
    return nn.Sequential(
        c1,
        nn.BatchNorm2d(mid),
        nn.LeakyReLU(),
        c2,
        nn.BatchNorm2d(outc),
        nn.LeakyReLU(),
        )

def block_up(inc,outc,mid_big=True,kern_size_desc=False):
    
    mid = 0
    if mid_big: mid=outc
    else: mid=inc
    
    if kern_size_desc:
        c1 = nn.ConvTranspose2d(inc,mid,4,2,1)
        c2 = nn.ConvTranspose2d(mid,outc,3,1,1)
        
    else:
        c1 = nn.ConvTranspose2d(inc,mid,3,1,1)
        c2 = nn.ConvTranspose2d(mid,outc,4,2,1)
        
    
    return nn.Sequential(
        c1,
        nn.BatchNorm2d(mid),
        nn.LeakyReLU(),
        c2,
        nn.BatchNorm2d(outc),
        nn.LeakyReLU(),
        )
        

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
#for img 3x16x16 pxl
class UNET_2_atten(nn.Module):
    def __init__(self,num_channels=3):
        super().__init__()
        
        self.from_img = nn.Conv2d(3,2**3,3,1,1)
        self.to_img = nn.ConvTranspose2d(2**3,num_channels,3,1,1)

        #encoder layers
        self.enc1 = nn.ModuleList([block_down(2**(i+1),2**(i+2)) for i in range(2,6+2)])
        #decoder layers
        self.ct2 = block_up(128*4,64*4)
        self.dec2 = nn.ModuleList([block_up(2**(i+3),2**(i+1)) for i in range(2,5+2)])
        # self.bottle_neck_down = nn.Linear(512,128)
        # self.bottle_neck_up = nn.Linear(128,512)
        self.skip = []
        self.soft = nn.Softmax()
        self.atten = nn.ModuleList([Attention_block(2**(i+2),2**(i+2),4) for i in range(2,5+2)])
        self.bottle_neck_down = nn.Linear(512,128)
        self.bottle_neck_up = nn.Linear(128,512)

    def enc(self, x):
        for i in self.enc1:
            x = i(x)
            self.skip.append(x)
        return x
        
    def dec(self, x):
        x = self.ct2(x)
        for k,(i,j) in enumerate(zip(self.dec2[::-1],self.atten[::-1])):
            d = j(x,self.skip[::-1][k+1])
            x = torch.cat((x,d),axis=1)
            x = i(x)
        return x

    def forward(self, x):
        self.skip = []
        x = self.from_img(x)

        x = self.enc(x)    

        x = self.bottle_neck_down(nn.Flatten()(x))
        x = self.bottle_neck_up(x)
        x = self.dec(x.resize(x.size()[0],512,1,1))    
        # x = self.dec(x)
        
        x = self.to_img(x)
        return self.soft(x)

        
#for img 3x16x16 pxl
class UNET_2_1(nn.Module):
    def __init__(self,num_channels=3):
        super().__init__()
        
        self.from_img = nn.Conv2d(3,2**5,3,1,1)
        self.to_img = nn.ConvTranspose2d(2**5,num_channels,3,1,1)

        #encoder layers
        self.enc1 = nn.ModuleList([block_down(2**(i+1),2**(i+2)) for i in range(2+2,6+2)])
        #decoder layers
        self.ct2 = block_up(128*4,64*4)
        self.dec2 = nn.ModuleList([block_up(2**(i+2),2**(i+1)) for i in range(2+2,5+2)])
        # self.bottle_neck_down = nn.Linear(512,128)
        # self.bottle_neck_up = nn.Linear(128,512)
        self.skip = []
        self.soft = nn.Softmax()

    def enc(self, x):
        for i in self.enc1:
            x = i(x)
            self.skip.append(x)
        return x
        
    def dec(self, x):
        x = self.ct2(x)
        for k,i in enumerate(self.dec2[::-1]):
            x += self.skip[::-1][k+1]
            x = i(x)
        return x

    def forward(self, x):
        self.skip = []
        x = self.from_img(x)

        x = self.enc(x)    

        # x = self.bottle_neck_down(nn.Flatten()(x))
        # x = self.bottle_neck_up(x)
        # x = self.dec(x.resize(x.size()[0],512,1,1))    
        x = self.dec(x)
        
        x = self.to_img(x)
        return self.soft(x)


#for img 3x32x32 pxl
class UNET_2_2(nn.Module):
    def __init__(self,num_channels=3,unet = UNET_2_1()):
        super().__init__()
        
        self.from_img = nn.Conv2d(3,2**4,3,1,1)
        self.to_img = nn.ConvTranspose2d(2**4,num_channels,3,1,1)

        #encoder layers
        self.enc1 =  unet.enc1[::-1]
        self.enc1.append(block_down(2**(4),2**(5)
                         ,mid_big=True, kern_size_desc=False))
        self.enc1 =  self.enc1[::-1]
        #decoder layers
        self.ct2 = block_up(128*4,64*4)
        # self.ct2 = block_up(128*2,64*2)
        self.dec2 =  unet.dec2[::-1]
        self.dec2.append(block_up(2**(5),2**(4)
                         ,mid_big=False, kern_size_desc=True))
        self.dec2 =  self.dec2[::-1]
                               
        #decoder layers
        self.skip = []
        self.soft = nn.Softmax()

    def enc(self, x):
        for i in self.enc1:
            x = i(x)
            self.skip.append(x)
        return x
        
    def dec(self, x):
        x = self.ct2(x)
        for k,i in enumerate(self.dec2[::-1]):
            x += self.skip[::-1][k+1]
            x = i(x)
        return x

    def forward(self, x):
        self.skip = []
        x = self.from_img(x)
        print(x.size())

        x = self.enc(x)    

        # x = self.bottle_neck_down(nn.Flatten()(x))
        # x = self.bottle_neck_up(x)
        # x = self.dec(x.resize(x.size()[0],512,1,1))    
        x = self.dec(x)
        
        x = self.to_img(x)
        return self.soft(x)
    
#for img 3x32x32 pxl
class UNET_2_3(nn.Module):
    def __init__(self,num_channels=3,unet = UNET_2_2()):
        super().__init__()
        
        self.from_img = nn.Conv2d(3,2**3,3,1,1)
        self.to_img = nn.ConvTranspose2d(2**3,num_channels,3,1,1)

        #encoder layers
        self.enc1 =  unet.enc1[::-1]
        self.enc1.append(block_down(2**(3),2**(4),mid_big=True, kern_size_desc=False))
        self.enc1 =  self.enc1[::-1]
        #decoder layers
        self.ct2 = block_up(128*4,64*4)
        # self.ct2 = block_up(128*2,64*2)
        self.dec2 =  unet.dec2[::-1]
        self.dec2.append(block_up(2**(4),2**(3),mid_big=False, kern_size_desc=True))
        self.dec2 =  self.dec2[::-1]
        self.bottle_neck_down = nn.Linear(512,128)
        self.bottle_neck_up = nn.Linear(128,512)

        #decoder layers
        self.skip = []
        self.soft = nn.Softmax()

    def enc(self, x):
        for i in self.enc1:
            x = i(x)
            self.skip.append(x)
        return x
        
    def dec(self, x):
        x = self.ct2(x)
        for k,i in enumerate(self.dec2[::-1]):
            x += self.skip[::-1][k+1]
            x = i(x)
        return x

    def forward(self, x):
        self.skip = []
        x = self.from_img(x)

        x = self.enc(x)    
        x = self.bottle_neck_down(nn.Flatten()(x))
        x = self.bottle_neck_up(x)
        x = self.dec(x.reshape(x.size()[0],512,1,1))
        
        x = self.to_img(x)
        return self.soft(x)
