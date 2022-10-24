import numpy as np
from model import FraxClassify
import torch

def data_loader():
    try:
        test_label = torch.from_numpy(np.load('data/mnist_test_Label.npy'))
        train_label = torch.from_numpy(np.load('data/mnist_train_Label.npy'))
        test_feat = torch.from_numpy(np.load('data/mnist_test_feat.npy'))
        train_feat = torch.from_numpy(np.load('data/mnist_train_feat.npy'))
        return test_label, train_label, test_feat, train_feat
    except Exception as e:
        print(e)
    
def cut_data(train_label, train_feat, test_label, test_feat, rank, world_size):
    data_len_min = len(train_feat) // world_size
    offset = len(train_feat) % world_size
    if rank < offset:
        start1 = rank*(data_len_min+1)
        end1 = start1+data_len_min+1
    else:
        start1 = offset*(data_len_min+1)+(rank-offset)*data_len_min
        end1 = start1+data_len_min
    data_len_min = len(test_feat) // world_size
    offset = len(test_feat) % world_size
    if rank < offset:
        start2 = rank*(data_len_min+1)
        end2 = start2+data_len_min+1
    else:
        start2 = offset*(data_len_min+1)+(rank-offset)*data_len_min
        end2 = start2+data_len_min
    
    return train_label[start1:end1], train_feat[start1:end1], test_label[start2:end2], test_feat[start2:end2]

def parallel_train(rank, world_size, layer_size, update_iter, measure_iter):
    print('I am ', rank)
    n_qubits = 6
    test_label, train_label, test_feat, train_feat = data_loader()
    train_label, train_feat, test_label, test_feat = cut_data(train_label, train_feat, test_label, test_feat, rank, world_size)
    model = FraxClassify(n_qubits, layer_size, measure_iter, world_size)
    for i in range(update_iter):
        model.fit(train=(train_feat, train_label))
        model.eval(train=(train_feat, train_label), test=(test_feat, test_label))
    model.get_accuracy()