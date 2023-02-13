# senior-thesis

This code is for IBM Quantum devices. You implement this code on *IBM Quantum Lab*.

If you want to implement this code, change the below code for your access.

> service = QiskitRuntimeService(<br>
&emsp;&emsp;channel='ibm_quantum',<br>
&emsp;&emsp;instance='ibm-q-utokyo/internal/cs-slecture8',<br>
)

You can change the experimental setting by the following code in *main.ipynb*.

> Q = 2 # qubit<br>
L = 2 # fraxis layer<br>
W = 63 # quantum node<br>
N = 63 # training data size<br>
M = 63 # evaluation data size<br>
U = 5 # update <br>
backend = service.backend('ibm_washington')
