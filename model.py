import torch
from circuit.embedding import amplitude_embedding
from circuit.ansatz import Frax_ansatz, replace_Frax_ansatz
import torch.distributed as dist
import math

def lastbit_Z(state):
    return torch.sum(state[0:len(state):2].abs()**2)-torch.sum(state[1:len(state):2].abs()**2)

class FraxClassify():
    def __init__(self, n_qubits, layer_size, measure_iter, world_size):
        self.n_qubits = n_qubits
        self.layer_size = layer_size
        self.measure_iter = measure_iter
        self.params = (torch.zeros(layer_size, n_qubits, 3) + 1/math.sqrt(3)).to(torch.complex64)
        self.world_size =world_size
        self.train_acc = []
        self.test_acc = []
        
    def fit(self, train):
        params = self.params
        train_feat, train_label = train
        x = amplitude_embedding(train_feat, self.n_qubits)
        for a in range(self.layer_size):
            for b in range(self.n_qubits):
                R = torch.zeros(3,3)
                for c in range(train_feat.shape[0]):
                    y = x[c]
                    for d in range(a):
                        y = Frax_ansatz(self.n_qubits, params[d]) @ y
                    rx = replace_Frax_ansatz(self.n_qubits, b, 'X', params[a]) @ y
                    ry = replace_Frax_ansatz(self.n_qubits, b, 'Y', params[a]) @ y
                    rz = replace_Frax_ansatz(self.n_qubits, b, 'Z', params[a]) @ y
                    rxy = replace_Frax_ansatz(self.n_qubits, b, 'XY', params[a]) @ y
                    rxz = replace_Frax_ansatz(self.n_qubits, b, 'XZ', params[a]) @ y
                    ryz = replace_Frax_ansatz(self.n_qubits, b, 'YZ', params[a]) @ y
                    for d in range(a+1, self.layer_size):
                        rx = Frax_ansatz(self.n_qubits, params[d]) @ rx
                        ry = Frax_ansatz(self.n_qubits, params[d]) @ ry       
                        rz = Frax_ansatz(self.n_qubits, params[d]) @ rz
                        rxy = Frax_ansatz(self.n_qubits, params[d]) @ rxy
                        rxz = Frax_ansatz(self.n_qubits, params[d]) @ rxz        
                        ryz = Frax_ansatz(self.n_qubits, params[d]) @ ryz
                        
                    rx = lastbit_Z(rx)
                    ry = lastbit_Z(ry)
                    rz = lastbit_Z(rz)
                    rxy = lastbit_Z(rxy)
                    rxz = lastbit_Z(rxz)
                    ryz = lastbit_Z(ryz)
                        
                    R[0,0] += train_label[c] * 2 * rx
                    R[0,1] += train_label[c] * (2 * rxy-rx-ry)
                    R[0,2] += train_label[c] * (2 * rxz-rx-rz)
                    R[1,1] += train_label[c] * 2 * ry
                    R[1,2] += train_label[c] * (2 * ryz-ry-rz)
                    R[2,1] += train_label[c] * 2 * rz
                    
                R[1,0] = R[0,1]
                R[2,0] = R[0,2]
                R[2,1] = R[1,2]
                group = dist.new_group(range(self.world_size))
                dist.all_reduce(R, op=dist.ReduceOp.SUM, group=group)
                if (dist.get_rank(group) == 0):
                    print(R)
                eigenvalues, eigenvectors = torch.linalg.eig(R)
                self.params[a, b] = eigenvectors[torch.argmin(eigenvalues.real)]
                self.params[a, b] /= torch.norm(self.params[a, b])
                
    def eval(self, train, test):
        test_score = 0
        train_score = 0
        train_feat, train_label = train
        test_feat, test_label = test
        train_size = train_label.shape[0]
        test_size = test_label.shape[0]
        for a in range(test_size):
            x = amplitude_embedding(test_feat[a], self.n_qubits)
            for b in range(self.layer_size):
                x = Frax_ansatz(self.n_qubits, self.params[b]) @ x
            test_score += test_label[a] * lastbit_Z(x)
        group = dist.new_group(range(self.world_size))
        dist.all_reduce(test_score, op=dist.ReduceOp.SUM, group=group)
        self.test_acc.append(test_score)
        
        for a in range(train_size):
            x = amplitude_embedding(train_feat[a], self.n_qubits)
            for b in range(self.layer_size):
                x = Frax_ansatz(self.n_qubits, self.params[b]) @ x
            train_score += train_label[a] * lastbit_Z(x)
        group = dist.new_group(range(self.world_size))
        dist.all_reduce(train_score, op=dist.ReduceOp.SUM, group=group)
        self.train_acc.append(train_score)
        
    def get_accuracy(self):
        if dist.get_rank() == 0:
            print(self.train_acc, self.test_acc)