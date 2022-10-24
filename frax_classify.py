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
    parser.add_argument('--W', type=int, default=1, help='Num of nodes, default=1')
    parser.add_argument('--L', type=int, default=4, help='Num of layers, default=4')
    parser.add_argument('--N', type=int, default=100, help='Num of updates, default=100')
    parser.add_argument('--M', type=int, default=100, help='Num of measurements, default=100')
    opt = parser.parse_args()
    return opt
    
def init_process(rank, world_size, layer_size, update_iter, measure_iter, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size, layer_size, update_iter, measure_iter)

if __name__ == '__main__':
    print("hello")
    opt = get_opt()
    processes = []
    st = time.time()
    for rank in range(opt.W):
        p = mp.Process(target=init_process, args=(rank, opt.W, opt.L, opt.N, opt.M, parallel_train))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print('Implementation time : ', time.time()-st)