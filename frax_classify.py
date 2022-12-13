'''
This code is for the implementation of the binary classification 
of '0' and '1', 8*8 MNIST.
Amplitude embedding is used for encoding, and 6 qubits are used.
'''

import argparse 
import torch.multiprocessing as mp
import os
from training import parallel_train
import torch.distributed as dist
import time

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Q', type=(int), help=('Num of qubits'))
    parser.add_argument('--L', type=int, help='Num of layers')
    parser.add_argument('--W', type=int, help='Num of nodes')
    parser.add_argument('--N', type=int, default=100, help='Num of training data')
    parser.add_argument('--M', type=int, default=100, help='Num of testing data')
    parser.add_argument('--U', type=int, default=20, help='Num of updates, default=20')
    opt = parser.parse_args()
    return opt
    
def init_process(rank, n_qubits, layer_size, world_size, num_train, num_test, update_iter, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, n_qubits, layer_size, world_size, num_train, num_test, update_iter)

if __name__ == '__main__':
    opt = get_opt()
    processes = []
    st = time.time()
    for rank in range(opt.W):
        p = mp.Process(target=init_process, args=(rank, opt.Q, opt.L, opt.W, opt.N, opt.M, opt.U, parallel_train))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print('Implementation time : ', time.time()-st)