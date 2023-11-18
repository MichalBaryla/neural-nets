import torch
import torch.nn as nn

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

##########################
### MODEL
##########################

class VAEX(torch.nn.Module):

    def __init__(self, latent_dim=100, ae_shape=None):
        
        super().__init__()
        self.latent_dim = latent_dim
        self.arr = [[2**i,2**(i+1)] for i in range(5,10)]

        if ae_shape==None:
            self.enc_shape = [[True,False][::-1]]*6
        else:
            self.enc_shape = ae_shape
            
        #encoder
        self.enc = nn.Sequential()
        for j,i in enumerate(self.arr):
            self.enc.add_module(
                module=block_down(i[0],i[1],self.enc_shape[j][0],self.enc_shape[j][1]),name=str(j))

        #decoder
        self.dec = nn.Sequential()
        for j,i in enumerate(self.arr[::-1]):
            self.dec.add_module(
                module=block_up(i[1],i[0],not(self.enc_shape[j][0]),not(self.enc_shape[j][1])),name=str(j))
        
            
            
        self.encoder = nn.Sequential(
            nn.Conv2d(3,self.arr[0][0],3,1,1),
            self.enc,
            nn.Conv2d(self.arr[-1][1],latent_dim,2,1,0),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,self.arr[-1][1],2,1,0),
            self.dec,
            nn.ConvTranspose2d(self.arr[0][0],3,3,1,1),
        )
        
        self.z_mean = torch.nn.Linear(latent_dim,latent_dim)
        self.z_log_var = torch.nn.Linear(latent_dim,latent_dim)

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to('cuda:0')
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0),1,1,x.size(1))
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        x = x.reshape(x.size(0),x.size(3),1,1)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(x)
        return encoded, z_mean, z_log_var, decoded

##########################
### MODEL
##########################

class VAEX_mini_dense(torch.nn.Module):

    def __init__(self, latent_dim=100, ae_shape=None, arr=None):
        
        super().__init__()
        self.latent_dim = latent_dim
        if arr==None:
            self.arr = [[2**i,2**(i+1)] for i in range(5,10)]
        else:
            self.arr = arr
            
        if ae_shape==None:
            self.enc_shape = [[True,False][::-1]]*6
        else:
            self.enc_shape = ae_shape
            
        #encoder
        self.enc = nn.Sequential()
        for j,i in enumerate(self.arr):
            self.enc.add_module(
                module=block_down(i[0],i[1],self.enc_shape[j][0],self.enc_shape[j][1]),name=str(j))

        #decoder
        self.dec = nn.Sequential()
        for j,i in enumerate(self.arr[::-1]):
            self.dec.add_module(
                module=block_up(i[1],i[0],not(self.enc_shape[j][0]),not(self.enc_shape[j][1])),name=str(j))

        self.lin_enc = nn.Sequential()
        for i in range(3):
            self.lin_enc.add_module(
                module = nn.Linear(
                    self.arr[-1*(i+1)][1],
                    self.arr[-1*(i+1)][1]//2
                ),
                name = 'ld_'+str(i)
            )            
                
        self.lin_dec = nn.Sequential()
        for i in range(3):
            self.lin_dec.add_module(
                module = nn.Linear(
                    self.arr[-1*(3-i)][1]//2,
                    self.arr[-1*(3-i)][1]
                ),
                name = 'le_'+str(i)
            )            
            
        self.encoder = nn.Sequential(
            nn.Conv2d(3,self.arr[0][0],3,1,1),
            self.enc,
            nn.Conv2d(self.arr[-1][1],self.arr[-1][1],2,1,0),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.arr[-1][1],self.arr[-1][1],2,1,0),
            self.dec,
            nn.ConvTranspose2d(self.arr[0][0],3,3,1,1),
        )
        
        self.z_mean = torch.nn.Linear(latent_dim,latent_dim)
        self.z_log_var = torch.nn.Linear(latent_dim,latent_dim)

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to('cuda:0')
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z
        
        
    def forward(self, x):
        x = self.encoder(x)

        x = x.reshape(x.size(0),1,1,x.size(1))
        x = self.lin_enc(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        x = self.lin_dec(x)
        x = x.reshape(x.size(0),x.size(3),1,1)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(x)
        return encoded, z_mean, z_log_var, decoded

##########################
### MODEL
##########################

class GANX(torch.nn.Module):

    def __init__(self, latent_dim=100, ae_shape=None):
        
        super().__init__()

        self.encoder = VAEX(latent_dim=latent_dim,ae_shape=ae_shape).encoder
        self.decoder = VAEX(latent_dim=latent_dim,ae_shape=ae_shape).decoder

        self.discriminator = nn.Sequential(
            self.encoder,
            nn.Flatten(),
            nn.Linear(100,1))
            
        self.generator = self.decoder
        
    def generator_forward(self, z):
        img = self.generator(z)
        return img
    
    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits

##########################
### MODEL
##########################

class VAEX_noiss(torch.nn.Module):

    def __init__(self, latent_dim=100, ae_shape=None):
        
        super().__init__()
        self.latent_dim = latent_dim
        # self.arr = [[2**i,2**(i+1)] for i in range(6,11)]
        self.arr = [[2**i,2**(i+1)] for i in range(5,10)]

        if ae_shape==None:
            self.enc_shape = [[True,False][::-1]]*6
        else:
            self.enc_shape = ae_shape
        # if dec_shape==None:
        #     self.dec_shape = [[False,True][::-1]]*4
        #encoder
        self.enc = nn.Sequential()
        for j,i in enumerate(self.arr):
            self.enc.add_module(
                module=block_down(i[0],i[1],self.enc_shape[j][0],self.enc_shape[j][1]),name=str(j))

        #decoder
        self.dec = nn.Sequential()
        for j,i in enumerate(self.arr[::-1]):
            self.dec.add_module(
                module=block_up(i[1],i[0],not(self.enc_shape[j][0]),not(self.enc_shape[j][1])),name=str(j))
        
            
            
        self.encoder = nn.Sequential(
            nn.Conv2d(3,self.arr[0][0],3,1,1),
            self.enc,
            nn.Conv2d(self.arr[-1][1],latent_dim,2,1,0),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,self.arr[-1][1],2,1,0),
            self.dec,
            nn.ConvTranspose2d(self.arr[0][0],3,3,1,1),
        )
        
        self.z_mean = torch.nn.Linear(latent_dim,latent_dim)
        # .to('cuda:0')
        self.z_log_var = torch.nn.Linear(latent_dim,latent_dim)
        # .to('cuda:0')

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to('cuda:0')
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0),1,1,x.size(1))
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        x = x.reshape(x.size(0),x.size(3),1,1)
        x = x + torch.randn(x.size()).to('cuda:0')/10
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(x)
        return encoded, z_mean, z_log_var, decoded

##########################
### MODEL
##########################

class VAE1(torch.nn.Module):

    def __init__(self, latent_dim=100, ae_shape=None):
        
        super().__init__()
        self.latent_dim = latent_dim
        self.arr = [[2**i,2**(i+1)] for i in range(6,11-2)]
        if ae_shape==None:
            self.enc_shape = [[True,False]]*4
        else:
            self.enc_shape = ae_shape
        # if dec_shape==None:
        #     self.dec_shape = [[False,True][::-1]]*4
        #encoder
        self.enc = nn.Sequential()
        for j,i in enumerate(self.arr):
            self.enc.add_module(
                module=block_down(i[0],i[1],self.enc_shape[j][0],self.enc_shape[j][1]),name=str(j))

        #decoder
        self.dec = nn.Sequential()
        for j,i in enumerate(self.arr[::-1]):
            self.dec.add_module(
                module=block_up(i[1],i[0],not(self.enc_shape[j][0]),not(self.enc_shape[j][1])),name=str(j))
        
            
            
        self.encoder = nn.Sequential(
            nn.Conv2d(3,self.arr[0][0],3,1,1),
            self.enc,
            nn.Conv2d(self.arr[-1][1],latent_dim,2,1,0),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,self.arr[-1][1],2,1,0),
            self.dec,
            nn.ConvTranspose2d(self.arr[0][0],3,3,1,1),
        )
        
        self.z_mean = torch.nn.Linear(latent_dim,latent_dim)
        # .to('cuda:0')
        self.z_log_var = torch.nn.Linear(latent_dim,latent_dim)
        # .to('cuda:0')

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to('cuda:0')
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0),1,1,x.size(1))
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        x = x.reshape(x.size(0),x.size(3),1,1)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(x)
        return encoded, z_mean, z_log_var, decoded

##########################
### MODEL
##########################

class VAE2(torch.nn.Module):

    def __init__(self, ae_shape=None, vae1 = None, latent_dim=100):
        
        super().__init__()
        if ae_shape==None:
            self.enc_shape = [[True,False]]
        else:
            self.enc_shape = ae_shape

        if vae1==None:
            self.vae1 = VAE1()
            
        else:
            self.vae1 = vae1

        self.arr = self.vae1.arr[0][0]
        self.arr1 = self.vae1.arr[-1][1]
        #encoder
        j=0
        self.enc = block_down(self.arr//2,self.arr,self.enc_shape[j][0],self.enc_shape[j][1])
        self.enc = nn.Sequential(
            self.enc,
            self.vae1.enc)
        #decoder
        self.dec = block_up(self.arr,self.arr//2,not(self.enc_shape[j][0]),not(self.enc_shape[j][1]))
        self.dec = nn.Sequential(
            self.vae1.dec,
            self.dec)
            
            
        self.encoder = nn.Sequential(
            nn.Conv2d(3,self.arr//2,3,1,1),
            self.enc,
            nn.Conv2d(self.arr1,latent_dim,2,1,0),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,self.arr1,2,1,0),
            self.dec,
            nn.ConvTranspose2d(self.arr//2,3,3,1,1),
        )
        
        self.z_mean = self.vae1.z_mean
        # .to('cuda:0')
        self.z_log_var = self.vae1.z_log_var 
        # .to('cuda:0')

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to('cuda:0')
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0),1,1,x.size(1))
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        x = x.reshape(x.size(0),x.size(3),1,1)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(x)
        return encoded, z_mean, z_log_var, decoded

##########################
### MODEL
##########################

class VAE3(torch.nn.Module):

    def __init__(self, ae_shape=None, vae1 = None, latent_dim=100):
        
        super().__init__()
        if ae_shape==None:
            self.enc_shape = [[True,False]]
        else:
            self.enc_shape = ae_shape

        if vae1==None:
            self.vae1 = VAE2()
            
        else:
            self.vae1 = vae1

        self.arr = self.vae1.arr//2
        self.arr1 = self.vae1.arr1
        #encoder
        j=0
        self.enc = block_down(self.arr//2,self.arr,self.enc_shape[j][0],self.enc_shape[j][1])

        #decoder
        self.dec = block_up(self.arr,self.arr//2,not(self.enc_shape[j][0]),not(self.enc_shape[j][1]))
            
            
        self.encoder = nn.Sequential(
            nn.Conv2d(3,self.arr//2,3,1,1),
            self.enc,
            self.vae1.enc,
            nn.Conv2d(self.arr1,latent_dim,2,1,0),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,self.arr1,2,1,0),
            self.vae1.dec,
            self.dec,
            nn.ConvTranspose2d(self.arr//2,3,3,1,1),
        )
        
        self.z_mean = self.vae1.z_mean
        # .to('cuda:0')
        self.z_log_var = self.vae1.z_log_var 
        # .to('cuda:0')

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to('cuda:0')
        z = z_mu + eps * torch.exp(z_log_var/2.).to('cuda:0')
        return z
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0),1,1,x.size(1))
        enc = x
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        x = x.reshape(x.size(0),x.size(3),1,1)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(x)
        return encoded, z_mean, z_log_var, decoded, enc
