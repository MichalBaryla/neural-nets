import os
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from torchvision import datasets, models, transforms
import torchvision 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torchvision import datasets, models, transforms

def train_vae_v1(num_epochs, model, optimizer, device, 
                 train_loader, loss_fn=None,
                 logging_interval=100, 
                 skip_epoch_stats=False,
                 reconstruction_term_weight=1,
                 save_model=None):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            features = features.to(device[0])

            # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(features)
            
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var) ,
                                      axis=1).to('cuda:0') # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension

            pixelwise = loss_fn(decoded, features.to('cuda:0'), reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean().to('cuda:0') # average over batch dimension
            loss = reconstruction_term_weight*pixelwise + kl_div
            
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_combined_per_epoch'].append(train_loss.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        fig, ax = plt.subplots(10,10, figsize=(64, 48))
        img = decoded.detach().cpu().numpy()
        img1 = features.detach().cpu().numpy()
        for i in range(5):
            for j in range(10):
                ax[j,i].imshow(img1[i+10*j].T)
                ax[j,i+5].imshow(img[i+10*j].T)
        plt.show()
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict

def train_gan_v1(num_epochs, model,
                 latent_dim, device, train_loader, loss_fn=None,
                 logging_interval=100, b=128,
                 save_model=None):
    print("training1")
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': [],
                'images_from_noise_per_epoch': []}
    
    torch.autograd.set_detect_anomaly(True)
    
    if loss_fn is None:
        loss_fn = F.binary_cross_entropy_with_logits

    # Batch of latent (noise) vectors for
    # evaluating / visualizing the training progress
    # of the generator
    lr = 0.0001
    start_time = time.time()
    fixed_noise = torch.randn(next(iter(train_loader))[0].size()[0], latent_dim, 1, 1,  device=device[0])  # format NCHW
    for epoch in range(num_epochs):
#         if(epoch<=int(num_epochs/2)-1):
#             lr+=0.00005
#         else:
#             lr-=0.00005
            
        optimizer_discr  = torch.optim.Adam(model.discriminator.parameters(),
                                       betas=(0.5, 0.999),
                                       lr=lr)

        optimizer_gen = torch.optim.Adam(model.generator.parameters(),
                                     betas=(0.5, 0.999),
                                     lr=lr)

        model.train()
        for batch_idx, (features,_) in enumerate(train_loader):
            batch_size = features.size(0)
            # batch_size = np.shape(next(iter(features))[1])

            # real images
            real_images = features.to(device[1])
            real_labels = torch.ones(batch_size, device=device[1]) # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim, 1, 1,  device=device[0])  # format NCHW
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size, device=device[1]) # fake label = 0
            flipped_fake_labels = real_labels # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.to(device[1])).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            
            discr_loss = 0.5*(real_loss + fake_loss)
            
            ###uncomment durring dicriminator trainning!!!
            
            discr_loss.backward(retain_graph=True)

            optimizer_discr.step()

            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

            # get discriminator loss on fake images with flipped labels
            discr_pred_fake = model.discriminator_forward(fake_images.to(device[1])).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward(retain_graph=True)

            optimizer_gen.step()
            
            # --------------------------
            # Logging
            # --------------------------   
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                       % (epoch+1, num_epochs, batch_idx, 
                          len(train_loader), gener_loss.item(), discr_loss.item()))
            
        ### Save images for evaluation
#         with torch.no_grad():
#             fake_images = model.generator_forward(fixed_noise).detach().cpu()
#             log_dict['images_from_noise_per_epoch'].append(
#                 torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
        fake_images = model.generator_forward(fixed_noise)
        y = model.discriminator_forward(fake_images.to('cuda:0')).detach().cpu().numpy()
        fake_images = fake_images.detach().cpu().numpy()
        print(len(y))
        print(batch_size)
        a = np.append(y.T, [np.arange(128).astype(np.int32)],axis=0).T.tolist()
        a.sort(key = lambda x: x[0])
        a = a[::-1]
        np.array(a).T[1].astype(np.int)
        aa = np.array(a).T[0].astype(np.float16)
        img1 = fake_images[np.array(a).T[1].astype(np.int)]
        fig, ax = plt.subplots(10,10, figsize=(64, 48))
        for i in range(10):
            for j in range(10):
                ax[j,i].imshow(img1[i+10*j].T)
                ax[j,i].set_title(aa[i+10*j].T)
        plt.show()
        print('lol')
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict

def train_vae_gan_v1(num_epochs, model,
                  device, train_loader, loss_fn=None,
                 logging_interval=100, b=128,
                 reconstruction_term_weight=1,
                 v_t_g=1, #vae to gan weight
                 save_model=None):
    print("training1")
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': [],
                'images_from_noise_per_epoch': []}
    
    torch.autograd.set_detect_anomaly(True)
    
    if loss_fn is None:
        loss_fn = F.binary_cross_entropy_with_logits

    # Batch of latent (noise) vectors for
    # evaluating / visualizing the training progress
    # of the generator
    lr = 0.0001
    start_time = time.time()
    for epoch in range(num_epochs):
#         if(epoch<=int(num_epochs/2)-1):
#             lr+=0.00005
#         else:
#             lr-=0.00005
            
        optimizer_discr  = torch.optim.Adam(model.discriminator.parameters(),
                                       betas=(0.5, 0.999),
                                       lr=lr)

        optimizer_gen = torch.optim.Adam(model.generator.parameters(),
                                     betas=(0.5, 0.999),
                                     lr=lr)

        model.train()
        for batch_idx, (features,_) in enumerate(train_loader):
            batch_size = features.size(0)
            # batch_size = np.shape(next(iter(features))[1])
            # real images
            real_images = features.to(device[1])
            real_labels = torch.ones(batch_size, device=device[1]) # real label = 1

            # generated (fake) images
            # noise = torch.randn(batch_size, latent_dim, 1, 1,  device=device[0])  # format NCHW
            encoded, z_mean, z_log_var,fake_images = model.generator_forward(features.to(device[0]))
            fake_labels = torch.zeros(batch_size, device=device[1]) # fake label = 0
            flipped_fake_labels = real_labels # here, fake label = 1
            
            # --------------------------
            # Train Discriminator
            # --------------------------
            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.to(device[1])).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            
            discr_loss = 0.5*(real_loss + fake_loss)
            
            ###uncomment durring discriminator trainning!!!
            
            discr_loss.backward(retain_graph=True)

            optimizer_discr.step()

            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                    - z_mean**2 
                                    - torch.exp(z_log_var), 
                                    axis=1).to(device[0]) # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension

            pixelwise = loss_fn(fake_images, features.to(device[0]), reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean().to(device[0]) # average over batch dimension
            loss = reconstruction_term_weight*pixelwise + kl_div
            

            # loss.backward(retain_graph=True)

            # UPDATE MODEL PARAMETERS
            # get discriminator loss on fake images with flipped labels

            discr_pred_fake = model.discriminator_forward(fake_images.to(device[1])).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)

            gener_loss.backward(retain_graph=True)

            optimizer_gen.step()
            
            # --------------------------
            # Logging
            # --------------------------   
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                       % (epoch+1, num_epochs, batch_idx, 
                          len(train_loader), gener_loss.item(), discr_loss.item()))
            
        ### Save images for evaluation
#         with torch.no_grad():
#             fake_images = model.generator_forward(fixed_noise).detach().cpu()
#             log_dict['images_from_noise_per_epoch'].append(
#                 torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        fig, ax = plt.subplots(10,10, figsize=(64, 48))
        img = fake_images.detach().cpu().numpy()
        img1 = features.detach().cpu().numpy()
        for i in range(5):
            for j in range(10):
                ax[j,i].imshow(img1[i+10*j].T)
                ax[j,i+5].imshow(img[i+10*j].T)
        plt.show()
        print('lol')
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict

def train_unet_1_1(num_epochs, model, optimizer, device, 
                 train_loader, loss_fn=None,
                 logging_interval=100,batchsize=128,
                 save_model=None):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            features = features.to(device[0])
        
            # FORWARD AND BACK PROP
            decoded = model(features)

            pixelwise = loss_fn(decoded, features.to('cuda:0'), reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            loss = pixelwise.mean().to('cuda:0') # average over batch dimension
            
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            # log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            # log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        fig, ax = plt.subplots(10,10, figsize=(64, 48))
        img = decoded.detach().cpu().numpy()
        img1 = features.detach().cpu().numpy()
        for i in range(5):
            for j in range(10):
                ax[j,i].imshow(img1[i+10*j].T)
                ax[j,i+5].imshow(img[i+10*j].T)
        plt.show()
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict
    
### metrics

def mtcs(out, y):
    bnr_conf_mtcs = []
    real_conf_mtcs = []
    o = torch.round(out)
    tp  = sum(o[y==1])/len(out)
    fp  = sum(o[y==0])/len(out)
    bnr_conf_mtcs.append([tp,fp])
    tp  = sum(out[y==1])/len(out)
    fp  = sum(out[y==0])/len(out)
    real_conf_mtcs.append([tp,fp])
    o -= 1
    o *= -1
    tn  = sum(o[y==0])/len(out)
    fn  = sum(o[y==1])/len(out)
    bnr_conf_mtcs.append([tn,fn])
    fn  = sum(out[y==0])/len(out)
    fn  = sum(out[y==1])/len(out)
    real_conf_mtcs.append([tn,fn])
    return bnr_conf_mtcs,real_conf_mtcs
    
### proper training for costume data loader

def train_smile_det(num_epochs, train_loader, test_loader,
                       labels, optimizer_discr, batch_size=None, 
                       model=None, device=['cuda:0','cuda:0'],
                       logging_interval=100,loss_fn=None):    
    
    if batch_size == None: 
        batch_size = 128

    if loss_fn is None:
        loss_fn = F.binary_cross_entropy_with_logits
        loss_plt = nn.BCEWithLogitsLoss(reduce=False)
    
    test = test_loader
    test_order = np.arange(len(test))
    for epoch in range(num_epochs):

        ###shuffling dataset
        
        order = np.arange(len(train_loader))
        np.random.shuffle(order)

        # temp_labels = torch.tensor(labels[order][:,31]).type(torch.float)
        # temp_train_loader = torch.tensor(train_loader[order]).type(torch.float)
        
        ### training loop

        model.train()
        for batch_idx , (features, _, y) in enumerate(train_loader):
            
            lbl = y.to(device[1])[:,31].type(torch.float)
            lbl += 1
            lbl /= 2

            fake_images = features.to('cuda:0')
            # emb = torch.tensor(temp_train_loader[batch_idx*batch_size : (batch_idx+1)*batch_size]).to(device[1])

            # --------------------------
            # Train Discriminator
            # --------------------------

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator(fake_images.to(device[1])).view(-1)
            fake_loss = loss_fn(discr_pred_fake, lbl)
            
            fake_loss.backward(retain_graph=True)

            optimizer_discr.step()
            optimizer_discr.zero_grad()         
            with torch.no_grad():
                if not batch_idx % logging_interval:
                    np.random.shuffle(test_order)
                    fake_imagess, _, lbll = next(iter(test_loader))
                    fake_imagess, lbll = fake_imagess.type(torch.float).to('cuda:0'), lbll.type(torch.float).to('cuda:0')[:,31]
                    discr_pred_fakee = model.discriminator(fake_imagess).view(-1)
                    test_loss = loss_fn(discr_pred_fakee, lbll)
                    
                    print('Epoch: %03d/%03d | Batch %03d/%03d | Train/Test Loss: %.4f/%.4f' 
                            % (epoch+1, num_epochs, batch_idx, len(train_loader)//128, 
                                fake_loss.item(), test_loss.item()))
                    print(mtcs(discr_pred_fakee,lbll))
                    print(mtcs(discr_pred_fake,lbl))
        
        with torch.no_grad():
            dics_loss_plt = loss_plt(discr_pred_fake, lbl)
            fig, ax = plt.subplots(10,10, figsize=(64//2, 48//2))
            for i in range(10):
                for j in range(10):
                    ax[j,i].imshow(fake_images[i+10*j].T.cpu().numpy())
                    # ax[j,i].set_title(lbl[i+10*j].T.cpu().numpy())
                    ax[j,i].set_title(torch.round((dics_loss_plt[i+10*j]+1)/2).T.cpu().numpy())
                    # ax[j,i].set_title((dics_loss_plt[i+10*j]).T.cpu().numpy())
            plt.show()
