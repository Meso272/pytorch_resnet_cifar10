import pickle 
import numpy as np
import os
datafolder="../cifar-10-batches-py"
train_data=np.zeros((50000,3,32,32),dtype=np.float32)
train_labels=np.zeros((50000),dtype=np.int64)
for i in range(1,6):
    datafile="data_batch_%d" % i
    datapath=os.path.join(datafolder,datafile)
    with open(datapath,"rb") as f:
        dct=pickle.load(f,encoding="bytes")
        #print(dct.keys())
        train_part=dct[b'data']
        label_part=dct[b'labels']
        train_part=train_part.reshape((-1,3,32,32))
        train_data[(i-1)*10000:i*10000]=train_part
        train_labels[(i-1)*10000:i*10000]=label_part

filtered_train=np.zeros((10,5000,3,32,32),dtype=np.float32)
filtered_counts=[0]*10
for i in range(50000):
    label=int(train_labels[i])
    idx=filtered_counts[label]
    filtered_train[label][idx]=train_data[i]
    filtered_counts[label]+=1
print(filtered_counts)
sampled_train=np.zeros((10000,3,32,32),dtype=np.float32)
sampled_labels=np.zeros((10000),dtype=np.int64)
for i in range(10):
    sample_idx=np.random.choice(5000,1000,replace=False)
    sampled_train[i*1000:(i+1)*1000]=filtered_train[i][sample_idx]
    #print(sampled_train)
    sampled_labels[i*1000:(i+1)*1000]=[i]*1000
#print(np.max(sampled_train))
#print(np.min(sampled_train))
test_data=np.zeros((10000,3,32,32),dtype=np.float32)
test_labels=np.zeros((10000),dtype=np.int64)
with open(os.path.join(datafolder,"test_batch"),"rb") as f:
    dct=pickle.load(f,encoding="bytes")
    test_data=dct[b'data'].reshape((-1,3,32,32)).astype(np.float32)

    test_labels=np.array(dct[b'labels'],dtype=np.int64)

sampled_train.tofile("../train_x_10000.dat")
sampled_labels.tofile("../train_y_10000.dat")
test_data.tofile("../test_x.dat")
test_labels.tofile("../test_y.dat")
    