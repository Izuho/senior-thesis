from circuit.ansatz import FraxisAnsatz, replace_FraxisAnsatz, FraxisFeatureMap
import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
#from qiskit.primitives import Sampler
from qiskit_ibm_runtime import Sampler
from qiskit.compiler import transpile


class FraxClassify():
    def __init__(self, n_qubits, layer_size, world_size, train_size, test_size, backend):
        self.n_qubits = n_qubits
        self.layer_size = layer_size
        self.world_size = world_size    
        self.params = np.zeros((layer_size, n_qubits, 3)) + np.array([0,0,1])
        self.train_size = train_size
        self.test_size = test_size
        self.backend = backend
        
    def fit_and_eval(self, X, y, X2, y2):
        sampler = Sampler()
        for a in range(self.layer_size):
            for b in range(self.n_qubits):
                R = np.zeros((3,3))
                for c in range(y.shape[0] // self.world_size):
                    qcs = []
                    for d in range(6):
                        qcs.append(QuantumCircuit(self.n_qubits * self.world_size, self.world_size))                  
                    feature_map = []
                    for d in range(self.world_size):
                        feature_map.append(FraxisFeatureMap(self.n_qubits, X[d + c * self.world_size]))  
                    for d in range(a):
                        original_ansatz = FraxisAnsatz(self.n_qubits, self.params[d])
                        for e in range(6):
                            for f in range(self.world_size):
                                qcs[e].compose(
                                    feature_map[f], 
                                    qubits=range(self.n_qubits * f, self.n_qubits * (f + 1), 1), 
                                    inplace=True
                                )
                                qcs[e].compose(
                                    original_ansatz, 
                                    qubits=range(self.n_qubits * f, self.n_qubits * (f + 1), 1), 
                                    inplace=True
                                )
                    for d in range(6):
                        for e in range(self.world_size):
                            qcs[d].compose(
                                feature_map[e], 
                                qubits=range(self.n_qubits * e, self.n_qubits * (e + 1), 1), 
                                inplace=True
                            )
                    ansatzs = replace_FraxisAnsatz(self.n_qubits, b, self.params[a])
                    for d in range(6):
                        for e in range(self.world_size):
                            qcs[d].compose(
                                ansatzs[d], 
                                qubits=range(self.n_qubits * e, self.n_qubits * (e + 1), 1), 
                                inplace=True
                            )
                    for d in range(a+1,self.layer_size,1):
                        original_ansatz = FraxisAnsatz(self.n_qubits, self.params[d])
                        for e in range(6):
                            for f in range(self.world_size):
                                qcs[e].compose(
                                    feature_map[f], 
                                    qubits=range(self.n_qubits * f, self.n_qubits * (f + 1), 1), 
                                    inplace=True
                                )
                                qcs[e].compose(
                                    original_ansatz, 
                                    qubits=range(self.n_qubits * f, self.n_qubits * (f + 1), 1), 
                                    inplace=True
                                )
                    for d in range(6):
                        qcs[d].measure(range(0, self.world_size * self.n_qubits, self.n_qubits), range(self.world_size))

                    transpiled_circuits = transpile(qcs, backend=self.backend)
                    
                    result = sampler.run(
                        circuits=transpiled_circuits
                    ).result().quasi_dists

                    r6s = np.zeros((6, self.world_size))
                    for d in range(6):
                        for e in result[d]:
                            for f in range(self.world_size):
                                if (e >> f) % 2 == 1:
                                    r6s[d, f] -= result[d][e]
                                else:
                                    r6s[d, f] += result[d][e]
              
                    R[0,0] += np.sum(y[c * self.world_size : (c + 1) * self.world_size : 1] * 2 * r6s[0])
                    R[0,1] += np.sum(y[c * self.world_size : (c + 1) * self.world_size : 1] * (2 * r6s[3] - r6s[0] - r6s[1]))
                    R[0,2] += np.sum(y[c * self.world_size : (c + 1) * self.world_size : 1] * (2 * r6s[4] - r6s[0] - r6s[2]))
                    R[1,1] += np.sum(y[c * self.world_size : (c + 1) * self.world_size : 1] * 2 * r6s[1])
                    R[1,2] += np.sum(y[c * self.world_size : (c + 1) * self.world_size : 1] * (2 * r6s[5] - r6s[1] - r6s[2]))
                    R[2,2] += np.sum(y[c * self.world_size : (c + 1) * self.world_size : 1] * 2 * r6s[2])


                R[1,0] = R[0,1]
                R[2,0] = R[0,2]
                R[2,1] = R[1,2]

                # Decrease the difference in behavior
                R = np.where(R*R<1e-8, 0, R) 
                #R /= self.train_size


                eigenvalues, eigenvectors = np.linalg.eigh(R)
                self.params[a, b] = eigenvectors[:, np.argmax(eigenvalues.real)]
                print(np.max(eigenvalues))

                acc_and_score = self.eval(X, y)

                print('ACC_train: ',acc_and_score[0],'\nSCORE_train: ',acc_and_score[1])

                acc_and_score = self.eval(X2, y2)
                    
                print('ACC_test: ',acc_and_score[0],'\nSCORE_test: ',acc_and_score[1])
        print(self.params)
        
    def eval(self, X, y):
        sampler = Sampler()
        acc_and_score = np.zeros(2)
        for a in range(y.shape[0]  // self.world_size):
            qc = QuantumCircuit(self.n_qubits * self.world_size, self.world_size)
            feature_map = []
            for b in range(self.world_size):
                feature_map.append(FraxisFeatureMap(self.n_qubits, X[b + a * self.world_size]))  
            for b in range(self.layer_size):
                original_ansatz = FraxisAnsatz(self.n_qubits, self.params[b])
                for c in range(self.world_size):
                    qc.compose(
                        feature_map[c], 
                        qubits=range(self.n_qubits * c, self.n_qubits * (c + 1), 1), 
                        inplace=True
                    )
                    qc.compose(
                        original_ansatz, 
                        qubits=range(self.n_qubits * c, self.n_qubits * (c + 1), 1), 
                        inplace=True
                    )
            
            qc.measure(range(0, self.n_qubits * self.world_size, self.n_qubits),range(self.world_size))
                    
            result = sampler.run(
                circuits=[qc],
            ).result().quasi_dists
            Zexp = np.zeros(self.world_size)
            for b in result[0]:
                for c in range(self.world_size):
                    if (b >> c) % 2 == 1:
                        Zexp[c] -= result[0][b]
                    else:
                        Zexp[c] += result[0][b]
  
            acc_and_score[0] += np.sum(np.where(y[a * self.world_size : (a + 1) * self.world_size : 1] * Zexp > 0, 1, 0))
            acc_and_score[1] += np.sum(y[a * self.world_size : (a + 1) * self.world_size : 1] * Zexp) * 2
        return acc_and_score