import torch
from circuit.ansatz import FraxisAnsatz, replace_FraxisAnsatz, FraxisFeatureMap
import torch.distributed as dist
import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.primitives import Sampler

class FraxClassify():
    def __init__(self, n_qubits, layer_size, world_size):
        self.n_qubits = n_qubits
        self.layer_size = layer_size
        self.world_size = world_size    
        self.params = np.zeros((layer_size, n_qubits, 2))
        
    def fit_and_eval(self, X,y,X2,y2):
        sampler = Sampler()
        for a in range(self.layer_size):
            for b in range(self.n_qubits):
                R = torch.zeros(3,3)
                for c in range(y.shape[0]):
                    qcs = []
                    feature_map = FraxisFeatureMap(self.n_qubits, X[c])
                    for d in range(6):
                        qcs.append(QuantumCircuit(self.n_qubits))
                    for d in range(a):
                        original_ansatz = FraxisAnsatz(self.n_qubits, self.params[d])
                        for e in range(6):
                            qcs[e].compose(
                                feature_map, 
                                qubits=range(self.n_qubits), 
                                inplace=True
                            )
                            qcs[e].compose(
                                original_ansatz, 
                                qubits=range(self.n_qubits), 
                                inplace=True
                            )
                    for d in range(6):
                        qcs[d].compose(
                            feature_map, 
                            qubits=range(self.n_qubits), 
                            inplace=True
                        )
                    ansatzs = replace_FraxisAnsatz(self.n_qubits, b, self.params[a])
                    for d in range(6):
                        qcs[d].compose(
                            ansatzs[d], 
                            qubits=range(self.n_qubits), 
                            inplace=True
                        )
                    for d in range(a+1,self.layer_size,1):
                        original_ansatz = FraxisAnsatz(self.n_qubits, self.params[d])
                        for e in range(6):
                            qcs[e].compose(
                                feature_map, 
                                qubits=range(self.n_qubits), 
                                inplace=True
                            )
                            qcs[e].compose(
                                original_ansatz, 
                                qubits=range(self.n_qubits), 
                                inplace=True
                            )
                    for d in range(6):
                        cr = ClassicalRegister(1)
                        qcs[d].add_register(cr)
                        qcs[d].measure(0,cr)
                    
                    result = sampler.run(
                        circuits=qcs
                    ).result().quasi_dists

                    r6s = np.zeros(6)
                    for d in range(6):
                        for bits in result[d]:
                            if bits == 1:
                                r6s[d] -= result[d][bits]
                            else:
                                r6s[d] += result[d][bits]
                                    
                    R[0,0] += np.sum(y[c] * 2 * r6s[0])
                    R[0,1] += np.sum(y[c] * (2 * r6s[3] - r6s[0] - r6s[1]))
                    R[0,2] += np.sum(y[c] * (2 * r6s[4] - r6s[0] - r6s[2]))
                    R[1,1] += np.sum(y[c] * 2 * r6s[1])
                    R[1,2] += np.sum(y[c] * (2 * r6s[5] - r6s[1] - r6s[2]))
                    R[2,2] += np.sum(y[c] * 2 * r6s[2])
                    
                R[1,0] = R[0,1]
                R[2,0] = R[0,2]
                R[2,1] = R[1,2]
                group = dist.new_group(range(self.world_size))
                dist.all_reduce(R, op=dist.ReduceOp.SUM, group=group)
                R = R.to('cpu').detach().numpy().copy()
                eigenvalues, eigenvectors = np.linalg.eigh(R)
                self.params[a, b] = eigenvectors[0:2, np.argmax(eigenvalues.real)]
                if dist.get_rank(group) == 0: print(np.max(eigenvalues))

                acc_and_score = self.eval(X,y,sampler)
                dist.all_reduce(acc_and_score, op=dist.ReduceOp.SUM, group=group)
                
                if abs(np.max(eigenvalues)-acc_and_score[1])>1e-4:
                    self.params[a,b,:] *= -1
                    acc_and_score = self.eval(X,y,sampler) 
                if dist.get_rank(group) == 0: print('ACC_train: ',acc_and_score[0],'\nSCORE_train: ',acc_and_score[1])
                
                acc_and_score = self.eval(X2,y2,sampler)
                
                if dist.get_rank(group) == 0: print('ACC_test: ',acc_and_score[0],'\nSCORE_test: ',acc_and_score[1])
        if dist.get_rank(group) == 0: print(self.params)
        
    def eval(self, X, y, sampler):
        acc_and_score = torch.zeros(2)
        for a in range(y.shape[0]):
            qc = QuantumCircuit(self.n_qubits)
            feature_map = FraxisFeatureMap(self.n_qubits, X[a])
            for b in range(self.layer_size):
                qc.compose(
                    feature_map, 
                    qubits=range(self.n_qubits), 
                    inplace=True
                )
                qc.compose(
                    FraxisAnsatz(self.n_qubits, self.params[b]), 
                    qubits=range(self.n_qubits), 
                    inplace=True
                )
            
            cr = ClassicalRegister(1)
            qc.add_register(cr)
            qc.measure(0,cr)
                    
            result = sampler.run(
                circuits=[qc],
            ).result().quasi_dists
            Zexp = 0
            for bits in result[0]:
                if bits == 1:
                    Zexp -= result[0][bits]
                else:
                    Zexp += result[0][bits]
                        
            acc_and_score[0] += np.sum(np.where(y[a]*Zexp>0,1,0))
            acc_and_score[1] += np.sum(y[a]*Zexp)*2
        return acc_and_score