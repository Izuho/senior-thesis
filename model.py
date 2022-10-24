import torch
from circuit.embedding import amplitude_embedding
from circuit.ansatz import Frax_ansatz, replace_Frax_ansatz
import torch.distributed as dist
import math

class FraxClassify():
    def __init__(self, n_qubits, layer_size, measure_iter, world_size):
        self.n_qubits = n_qubits
        self.layer_size = layer_size
        self.measure_iter = measure_iter
        self.params = (torch.zeros(layer_size, n_qubits, 3) + 1/math.sqrt(3)).to(torch.complex64)
        self.R = torch.zeros(3, 3)
        self.world_size =world_size
        self.train_acc = []
        self.test_acc = []
        
    def fit(self, train):
        params = self.params
        train_feat, train_label = train
        x = amplitude_embedding(train_feat, self.n_qubits)
        for a in range(self.layer_size):
            for b in range(self.n_qubits):
                for c in range(train_feat.shape[0]):
                    for d in range(a):
                        x[c] = Frax_ansatz(self.n_qubits, params[d]) @ x[c]
                    rx = replace_Frax_ansatz(self.n_qubits, b, 'X', params[a]) @ x[c]
                    ry = replace_Frax_ansatz(self.n_qubits, b, 'Y', params[a]) @ x[c]
                    rz = replace_Frax_ansatz(self.n_qubits, b, 'Z', params[a]) @ x[c]
                    rxy = replace_Frax_ansatz(self.n_qubits, b, 'XY', params[a]) @ x[c]
                    rxz = replace_Frax_ansatz(self.n_qubits, b, 'XZ', params[a]) @ x[c]
                    ryz = replace_Frax_ansatz(self.n_qubits, b, 'YZ', params[a]) @ x[c]
                    for d in range(a+1, self.layer_size):
                        rx = Frax_ansatz(self.n_qubits, params[d]) @ rx        
                        ry = Frax_ansatz(self.n_qubits, params[d]) @ ry       
                        rz = Frax_ansatz(self.n_qubits, params[d]) @ rz
                        rxy = Frax_ansatz(self.n_qubits, params[d]) @ rxy
                        rxz = Frax_ansatz(self.n_qubits, params[d]) @ rxz        
                        ryz = Frax_ansatz(self.n_qubits, params[d]) @ ryz
                        
                    rx = torch.sum(rx[0:len(rx):2].abs()**2)-torch.sum(rx[1:len(rx):2].abs()**2)
                    ry = torch.sum(ry[0:len(ry):2].abs()**2)-torch.sum(ry[1:len(ry):2].abs()**2)
                    rz = torch.sum(rz[0:len(rz):2].abs()**2)-torch.sum(rz[1:len(rz):2].abs()**2)
                    rxy = torch.sum(rxy[0:len(rxy):2].abs()**2)-torch.sum(rxy[1:len(rxy):2].abs()**2)
                    rxz = torch.sum(rxz[0:len(rxz):2].abs()**2)-torch.sum(rxz[1:len(rxz):2].abs()**2)
                    ryz = torch.sum(ryz[0:len(ryz):2].abs()**2)-torch.sum(ryz[1:len(ryz):2].abs()**2)
                        
                    self.R[0,0] += train_label[c] * 2 * rx
                    self.R[0,1] += train_label[c] * (2 * rxy-rx-ry)
                    self.R[0,2] += train_label[c] * (2 * rxz-rx-rz)
                    self.R[1,1] += train_label[c] * 2 * ry
                    self.R[1,2] += train_label[c] * (2 * ryz-ry-rz)
                    self.R[2,1] += train_label[c] * 2 * rz
                    
                self.R[1,0] = self.R[0,1]
                self.R[2,0] = self.R[0,2]
                self.R[2,1] = self.R[1,2]
                group = dist.new_group(range(self.world_size))
                dist.all_reduce(self.R, op=dist.ReduceOp.SUM, group=group)
                eigenvalues, eigenvectors = torch.linalg.eig(self.R)
                params[a, b] = eigenvectors[torch.argmin(eigenvalues.real)]
                
    def eval(self, train, test):
        test_score = 0
        train_score = 0
        train_feat, train_label = train
        test_feat, test_label = test
        train_size = train_label.shape[0]
        test_size = test_label.shape[0]
        for a in range(test_size):
            x = amplitude_embedding(test_feat[a], self.n_qubits)
            for a in range(self.layer_size):
                x = Frax_ansatz(self.n_qubits, self.params[a]) @ x
            test_score += test_label[a] * (torch.sum(x[0:len(x):2].abs()**2)-torch.sum(x[1:len(x):2].abs()**2))
        group = dist.new_group(range(self.world_size))
        dist.all_reduce(test_score, op=dist.ReduceOp.SUM, group=group)
        self.test_acc.append(test_score)
        
        for a in range(train_size):
            x = amplitude_embedding(train_feat[a], self.n_qubits)
            for a in range(self.layer_size):
                x = Frax_ansatz(self.n_qubits, self.params[a]) @ x
            train_score += train_label[a] * (torch.sum(x[0:len(x):2].abs()**2)-torch.sum(x[1:len(x):2].abs()**2))
        group = dist.new_group(range(self.world_size))
        dist.all_reduce(train_score, op=dist.ReduceOp.SUM, group=group)
        self.train_acc.append(train_score)
        
    def get_accuracy(self):
        if dist.get_rank() == 0:
            print(self.train_acc, self.test_acc)