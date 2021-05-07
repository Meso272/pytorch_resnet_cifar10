
import numpy as np 
import torch
from torch.utils.data import *


class cifar10(Dataset):
    def __init__(self,images,labels):
        self.images=np.fromfile(images,dtype=np.float32).reshape((-1,3,32,32))
        self.labels=np.fromfile(labels,dtype=np.int64)
        self.len=self.images.shape[0]
        print(np.max(self.images))
        for i in range(self.len):
            #print(self.images[i])

            self.images[i]=(self.images[i]-np.mean(self.images[i]))/((1e-6)+np.std(self.images[i]))
        '''
        if torch.cuda.is_available():
            self.images=torch.Tensor(self.images).to('cuda')
            self.labels=torch.Tensor(self.labels).to('cuda')
        '''
        
    def __len__(self):
        return self.len
    def __getitem__(self,idx):
        return self.images[idx],self.labels[idx]
