# Job launching program  for 3 to 11 Qubit code on IBM Cairo.
# You can input the size of the code (number of data qubits) and the initial state (unencoded '1' or encoded |+>).
# 4 test qubit will be initialized in the same state and left idling during the code.
# The program automatically selects the best initial layout for the code on Cairo constructing a continuous chain that avoids the most noisy qubits.

#import qiskit
import matplotlib.pyplot as plt
import numpy as np
import math
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit import QuantumRegister, ClassicalRegister
import qiskit.quantum_info as qi
from qiskit_aer import AerSimulator
from typing import List, Optional
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error    # Import from Qiskit Aer noise module
from itertools import permutations, combinations
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_provider import IBMProvider

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

provider = IBMProvider(instance='ibm-q-cern/infn/qcqcd1')

n = input("Repetition code on IBM Cairo\nInsert code size (number of data qubits) \n")
bs = input("Insert '1' if you want to store an unencoded '1', input '2' if you want to encode and correct a |+> state\n")
n = int(n)
bs = int(bs)

if n==11:
    initial_layout =  [6, 7, 4, 1, 2, 3, 5, 8, 11, 14, 13, 12, 15, 18, 21, 23, 24, 25, 22, 19, 20, 0, 9, 26, 17]    #11 qubit code
if n==9:
    initial_layout = [6, 7, 4, 1, 2, 3, 5, 8, 11, 14, 13, 12, 15, 18, 21, 23, 24, 0, 9, 20, 17]    #9 qubit code
if n==7:
    initial_layout = [6, 7, 4, 1, 2, 3, 5, 8, 11, 14, 13, 12, 15, 0, 9, 20, 17]    #7 qubit code
if n==5:
    initial_layout = [6, 7, 4, 1, 2, 3, 5, 8, 11,  0,  9, 20, 17]  #5 qubit code
if n==3:
    initial_layout = [3, 5, 8, 11, 14, 6, 17, 26, 19]  #3 qubit code

#LOOKUPTABLE DEFINITION
nerr = int((n-1)/2)     #maximum number of errors that can be corrected
nsyn = 0
for i in range (nerr):
    nsyn = nsyn + math.comb(n,i+1)

def gen(n):
    if n == 1:
        yield from [[0], [1]]
    else:
        for a in gen(n-1):
            yield [0] + a
        
        for a in gen(n-1):
            yield [1] + a
            
cs = sorted([
    c
    for c in gen(n)
    if sum(c) <= nerr
], key=lambda x : sum(x))

cs.pop(0)

ancmeas = np.zeros((n-1), dtype=int)
syndrome = np.zeros((nsyn), dtype=int)
positions2 = [np.zeros((math.comb(n,x+1),x+1), dtype=int) for x in range(nerr)]

ind=0
isub = 0
ierr = 0
for x in cs:

    indpos = 0
    if (ierr!=sum(x)-1):
        isub = isub + math.comb(n,ierr+1)
        ierr = sum(x)-1

    ancmeas.fill(0) 
    for j in range (n-1,0,-1):
        if(x[j]!=x[j-1]):   #if two consecutive qubits are different, the ancilla measurement is 1 (qiskit reads measurements in reverse order)
            ancmeas[n-1-j] = 1
        if(x[j]==1):
            positions2[ierr][ind-isub][indpos]=j  #saves the positions of the errors
            indpos=indpos+1
    if(x[0]==1):
        positions2[ierr][ind-isub][indpos]=0    
    syndrome[ind] = int(''.join(map(str, ancmeas)), 2)  #salva la sindrome in decimale
    ind=ind+1

positions = []
[positions.extend(list(l)) for l in positions2]
#syndrome is a list of all possible syndrome measurements (in decimal and reversed because qiskit measures in reverse order)    syndrome[0:n_syndromes]
#positions is a list of all the corresponding positions of errors for each possible syndrome     positions[0:n_syndromes][0:n_faulty_qubits]

print("\nstep 0 done\n")

#Initialization of the quantum registers: 3 data qubits, 2 ancillas, 2 classical bits

qr = QuantumRegister(2*n+3)
sbit = ClassicalRegister(n-1, 'syndrome_bit')
mbit = ClassicalRegister(n, 'measure_bit')
tbit = ClassicalRegister(4, 'test_bit')
qc = QuantumCircuit(qr, sbit, mbit, tbit)

data = [i for i in range(0, 2*n-1, 2)]
anc = [i for i in range(1, 2*n-2, 2)]
test = [2*n-1, 2*n, 2*n+1, 2*n+2]

if(bs==2):
    #Save |+> with smallest depth encoding
    qc.h(qr[test])
    qc.h(qr[data[0]])

    for i in range (2*n-2):
        qc.cx(qr[i], qr[i+1])

    for i in range (2*n-2, 1, -2):
        qc.cx(qr[i], qr[i-1])


if(bs==1):
    #OR save |1> without encoding
    qc.x(qr[test])
    qc.x(qr[data])
    #entangle n-1 couples of data-qubit to n-1 ancillas to study internal correlation in the logical qubit
    for j in range (n-1):
        qc.cx(qr[data[j]], anc[j])
        qc.cx(qr[data[j+1]], anc[j])



# ancilla_i shows if data_qubit_i = data_qubit_i+1

#measure the ancillas to detect errors
for j in range (n-1):
    qc.measure(qr[anc[j]], sbit[j])  #ancilla is measured and the result stored in a classical bit

#if an error is detected in one of the qubits we apply a classically controlled not gate to correct it
for i_syn in range (len(syndrome)):
    with qc.if_test((sbit, int(syndrome[i_syn]))):
        for j in range (len(positions[i_syn])):
            qc.x(qr[data[positions[i_syn][j]]])

#    print("\nstep {} out of {} done \n".format(i+1, rs))

for i in range (n):
    qc.measure(qr[data[i]], mbit[i])
for i in range (4):
    qc.measure(qr[test[i]], tbit[i])

backend = provider.get_backend("ibm_cairo")
transpiled = transpile(qc, backend=backend, initial_layout=initial_layout)

job = backend.run(
    transpiled,
    dynamic=True,
    shots=5120,
)

print(job.status())
print(job.job_id())

