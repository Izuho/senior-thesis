from qiskit.circuit import QuantumCircuit
import numpy as np

def fraxis_gate(nx, ny, nz=None):
    nx, ny, nz = _validate(nx, ny, nz)
    theta = 2*(np.arccos(nz))
    phi = None
    if nx==0:
      phi=np.pi/2
    else:
      phi=np.arctan2(ny, nx)
    circ = QuantumCircuit(1)
    
    circ.u(theta,phi,np.pi-phi,[0])
    return circ

def _validate(nx, ny, nz=None):
    if nz == None:
        nx = nx.real
        ny = ny.real
        nz2 = 1-nx**2.-ny**2.
        nz = np.sqrt(nz2).real if nz2 > 0 else 0
    return nx, ny, nz

def FraxisFeatureMap(num_qubits, data):
    circ = QuantumCircuit(num_qubits)
    for i in range(
        data.shape[0]//2
    ):
        circ.compose(fraxis_gate(data[2*i], data[2*i+1]), qubits=[i%num_qubits], inplace=True)
        if (i+1)%num_qubits == 0:
            for j in range(0,num_qubits,2):
                if j+1 < num_qubits:
                    circ.cz(j,j+1)
            for j in range(1,num_qubits,2):
                if j+1 < num_qubits:
                    circ.cz(j,j+1)        
    return circ

def FraxisAnsatz(num_qubits, params):
    circ = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        circ.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
    for j in range(0,num_qubits,2):
        if j+1 < num_qubits:
            circ.cz(j,j+1)
    for j in range(1,num_qubits,2):
        if j+1 < num_qubits:
            circ.cz(j,j+1)        
    return circ

def replace_FraxisAnsatz(num_qubits, target, params):
    circX = QuantumCircuit(num_qubits)
    circY = QuantumCircuit(num_qubits)
    circZ = QuantumCircuit(num_qubits)
    circXY = QuantumCircuit(num_qubits)
    circXZ = QuantumCircuit(num_qubits)
    circYZ = QuantumCircuit(num_qubits)
    for i in range(target):
        circX.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
        circY.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
        circZ.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
        circXY.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
        circXZ.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
        circYZ.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
    circX.x(target)
    circY.y(target)
    circZ.z(target)
    circXY.u(np.pi, np.pi*0.25, np.pi*0.75, target)
    circXZ.u(np.pi*0.5, 0, np.pi, target)
    circYZ.u(np.pi*0.5, np.pi*0.5, np.pi*0.5, target)
    for i in range(target+1, num_qubits, 1):
        circX.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
        circY.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
        circZ.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
        circXY.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
        circXZ.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
        circYZ.compose(fraxis_gate(params[i,0], params[i,1]), qubits=[i], inplace=True)
    for j in range(0,num_qubits,2):
        if j+1 < num_qubits:
            circX.cz(j,j+1)
            circY.cz(j,j+1)
            circZ.cz(j,j+1)
            circXY.cz(j,j+1)
            circXZ.cz(j,j+1)
            circYZ.cz(j,j+1)
    for j in range(1,num_qubits,2):
        if j+1 < num_qubits:
            circX.cz(j,j+1)
            circY.cz(j,j+1)
            circZ.cz(j,j+1)
            circXY.cz(j,j+1)
            circXZ.cz(j,j+1)
            circYZ.cz(j,j+1)
    return [circX, circY, circZ, circXY, circXZ, circYZ]