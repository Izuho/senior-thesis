from .basis_gate import Frax, CZ, kronecker, I, X, Y, Z
import torch
from math import sqrt

def CZ_layer(n_qubits):
    if n_qubits == 2:
        return CZ
    gate1 = CZ
    for i in range(2, n_qubits, 2):
        if i+1 < n_qubits:
            gate1 = kronecker(gate1, CZ)
        else:
            gate1 = kronecker(gate1, I)
    gate2 = CZ
    gate2 = kronecker(I, gate2)
    for i in range(3, n_qubits, 2):
        if i+1 < n_qubits:
            gate2 = kronecker(gate2, CZ)
        else:
            gate2 = kronecker(gate2, I)
    return torch.mm(gate2, gate1)

def Frax_ansatz(n_qubits, param):
    # param : torch.Tensor of (n_qubits, 3)
    x = 1
    for i in range(n_qubits):
        x = kronecker(x, Frax(param[i]))
    return torch.mm(CZ_layer(n_qubits), x)

def replace_Frax_ansatz(n_qubits, measured_qubit, observable, param):
    x = 1
    for i in range(measured_qubit):
        x = kronecker(x, Frax(param[i]))
        
    if observable == 'X':
        x = kronecker(x, X)
    elif observable == 'Y':
        x = kronecker(x, Y)
    elif observable == 'Z':
        x = kronecker(x, Z)
    elif observable == 'XY':
        x = kronecker(x, (X+Y)/sqrt(2))
    elif observable == 'XZ':
        x = kronecker(x, (X+Z)/sqrt(2))
    elif observable == 'YZ':
        x = kronecker(x, (Y+Z)/sqrt(2))

    for i in range(measured_qubit+1, n_qubits):
        x = kronecker(x, Frax(param[i]))
    return torch.mm(CZ_layer(n_qubits), x)