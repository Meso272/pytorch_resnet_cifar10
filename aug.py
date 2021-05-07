import torch
import numpy as np

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lmd = np.random.beta(alpha, alpha)
    else:
        lmd = 1

    batch_size = x.size()[0]
    if torch.cuda.is_available():
        shuffled_idx = torch.randperm(batch_size).cuda()
    else:
        shuffled_idx = torch.randperm(batch_size)

    x_mix = lmd * x + (1 - lmd) * x[shuffled_idx, :]
    y_1, y_2 = y, y[shuffled_idx]
    return x_mix, y_1, y_2, lmd


def mixup_criterion(criterion, pred, y_1, y_2, lmd):
    return lmd * criterion(pred, y_1) + (1 - lmd) * criterion(pred, y_2)


def cutoff(x,K):
    batch_size=x.size()[0]
    channel_num=x.size()[1]
    h=x.size()[2]
    w=x.size()[3]
    #print(batch_size,channel_num,h,w)
    for i in range(batch_size):
        if np.random.choice(2):
            continue
        half_size=K//2

        start_x=np.random.choice(h)-half_size
        start_y=np.random.choice(w)-half_size
        end_x=start_x+K
        end_y=start_y+K
        start_x=max(0,start_x)
        start_y=max(0,start_y)
        end_x=min(h,end_x)
        end_y=min(w,end_y)
        zero=torch.zeros((channel_num,end_x-start_x,end_y-start_y))
        if torch.cuda.is_available():
            zero=zero.cuda()
        x[i,:,start_x:end_x,start_y:end_y]=zero



    return x


def standard(x,K):
    batch_size=x.size()[0]
    channel_num=x.size()[1]
    h=x.size()[2]
    w=x.size()[3]
    for i in range(batch_size):
        h_shift=np.random.choice(np.arange(-K,K+1))
        w_shift=np.random.choice(np.arange(-K,K+1))
        x[i]=torch.roll(x[i],[h_shift,w_shift],dims=[-2,-1])

        if h_shift>=0:
            a=0
            b=h_shift
        else:
            a=h+h_shift
            b=h
        if w_shift>=0:
            c=0
            d=w_shift
        else:
            c=w+w_shift
            d=w
        zero=torch.zeros((channel_num,b-a,d-c))
        if torch.cuda.is_available():
            zero=zero.cuda()
        x[i,:,a:b,c:d]=zero
        if np.random.choice(2):
            x[i]=torch.flip(x[i],[-1])
    return x


