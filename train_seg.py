from helper_seg import *



def train_unet_M(num_epochs, model, optimizer, device, 
                 train_loader, loss_fn=None,
                 logging_interval=100,batchsize=128,
                 save_model=None,two_chan=True):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn is None:
        loss_fn = F.binary_cross_entropy

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, y) in enumerate(train_loader):

            features = features.to(device[0])
            y = y.to(device[0]).type(torch.float)
            if two_chan==True:
                y += 1
                y /= 2
                a = y.clone()
                a *= -1
                a +=  1
                y = torch.round(torch.cat([a,y],axis=1))
                
            # FORWARD AND BACK PROP
            decoded = model(features)

            pixelwise = loss_fn(decoded, y.to('cuda:0'), reduction='none')
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
        img = decoded.detach().cpu()
        # .numpy()
        img1 = features.detach().cpu().numpy()
        for i in range(5):
            for j in range(10):
                ax[j,i].imshow(img1[i+10*j].T)
                ax[j,i+5].imshow(vis_iou(y[i+10*j][[0]],img[i+10*j][[0]]).T)
        plt.show()
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict
def train_unet_M_1(num_epochs, model, optimizer, device, 
                 train_loader, test_loader, a, loss_fn=None,
                 logging_interval=100,batchsize=128,
                 save_model=None,two_chan=True):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn is None:
        loss_fn = F.binary_cross_entropy

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, y) in enumerate(train_loader):

            features = features.to(device[0])
            y = y.to(device[0]).type(torch.float)
            y = seg_label_smooth(y,a)
            y = torch.cat([y,1-y],axis=1)
            
            # FORWARD AND BACK PROP
            decoded = model(features)

            pixelwise = loss_fn(decoded, y.to('cuda:0'), reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            loss = pixelwise.mean().to('cuda:0') # average over batch dimension
            
            optimizer.zero_grad()
            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            if not batch_idx % logging_interval:
                tr_iou = iou(decoded[:,0,:,:],y[:,0,:,:])
                tr_riou = real_iou(decoded[:,0,:,:],y[:,0,:,:])
                with torch.no_grad():
                    features, y = next(iter(test_loader))
                    features = features.to(device[0])
                    y = y.to(device[0]).type(torch.float)

                    y = seg_label_smooth(y,a)
                    y = torch.cat([y,1-y],axis=1)
                    
                    # FORWARD AND BACK PROP
                    decoded = model(features)

                    pixelwise = loss_fn(decoded, y.to('cuda:0'), reduction='none')
                    pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
                    test_loss = pixelwise.mean().to('cuda:0') # average over batch dimension

                    ts_iou = iou(decoded[:,0,:,:],y[:,0,:,:])
                    ts_riou = real_iou(decoded[:,0,:,:],y[:,0,:,:])
                    
                print('Epoch: %03d/%03d | Batch %04d/%04d | Train Loss/iou/riou/: %.4f/%.4f/%.4f/ | Test Loss/iou/riou/: %.4f/%.4f/%.4f/'
                      % (epoch+1, num_epochs, batch_idx, len(train_loader), 
                         loss, tr_iou, tr_riou,
                         test_loss, ts_iou, ts_riou,))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        fig, ax = plt.subplots(10,10, figsize=(64, 48))
        img = decoded.detach().cpu()
        # .numpy()
        img1 = features.detach().cpu().numpy()
        for i in range(5):
            for j in range(10):
                ax[j,i].imshow(img1[i+10*j].T)
                ax[j,i+5].imshow(vis_iou(y[i+10*j][[0]],img[i+10*j][[0]]).T)
        plt.show()
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict
def train_unet_M_noiss_11(num_epochs, model, optimizer, device, 
                        train_loader, loss_fn=None,
                        logging_interval=100,batchsize=128,
                        save_model=None,two_chan=True):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn is None:
        loss_fn = F.binary_cross_entropy

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, y) in enumerate(train_loader):

            features = features.to(device[0])
            features += torch.randn(features.size()).to(device[0])
            y = y.to(device[0]).type(torch.float)
            if two_chan==True:
                a = y.clone()
                a *= -1
                a +=  1
                y = torch.cat([a,y],axis=1)
                
            # FORWARD AND BACK PROP
            decoded = model(features)

            pixelwise = loss_fn(decoded, y.to('cuda:0'), reduction='none')
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
        img = decoded.detach().cpu()
        # .numpy()
        img1 = features.detach().cpu().numpy()
        for i in range(5):
            for j in range(10):
                ax[j,i].imshow(img1[i+10*j].T)
                ax[j,i+5].imshow(vis_iou(y[i+10*j][[0]],img[i+10*j][[0]]).T)
        plt.show()
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict