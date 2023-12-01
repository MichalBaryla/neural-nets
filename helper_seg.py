import torch
import torch.nn.functional as F


def vis_iou(y,out):
    sum = y.to('cpu')+torch.round(out.to('cpu'))
    return  torch.cat([
    y.to('cpu'),
    torch.round(out.to('cpu')),
    (sum==2).type(torch.float),])

def iou(y,out):
    yy = torch.round(y)+ torch.round(out)
    return yy[yy==2].sum()/yy[yy>=1].sum()

def real_iou(y,out):
    t = (y*out).sum()
    return t/((y+out).sum()-t)

def center_crop(img, sz):
    s = torch.tensor(img.size()[-2:])
    ss = s.type(torch.float) - sz
    ss /= 2
    ss = ss.type(torch.int)
    return img[:,:,
        ss[0]:ss[0]+sz[0],
        ss[1]:ss[1]+sz[1]]

def seg_label_smooth(im, a):
    ### working only for square images
    ### may fix it for rectangle latter if would be needed
    s = im.size(-1)
    s1=int(round(s*1.1,0))
    s2=int(round(s1*1.2,0))
    s2=int(round(s2*1.3,0))
    # im = im.reshape(1,1,s,s)
    yy1 = F.interpolate(im, size = (s1,s1))
    yy2 = F.interpolate(yy1, size = (s2,s2))
    yy3 = F.interpolate(yy1, size = (s2,s2))

    return (a**(im+ center_crop(yy1,torch.tensor([64,64]))+\
                    center_crop(yy2,torch.tensor([64,64]))+\
                    center_crop(yy3,torch.tensor([64,64])))-1)/a**4