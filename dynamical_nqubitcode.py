# Cyclic n-Qubit Repetition code simulation with bitflip and/or depolarizing noise model.
# You can input the size of the code, the number of qec rounds, the wait time between rounds and the noise model parameters (error probabilities).
# The output of the program is the success probability of the code in those conditions. Also, it prints the theoretical probability of a single
# qubit surviving unaltered the same total time in a bitflip or depolarizing memory with equal error rates to the noise model.

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


def get_noise(p_bitflip, p_gate):   #returns a noise model with p_bitflip = probability of bitflip and p_gate = probability of depolarizing gate error
    bit_flip = pauli_error([('X', p_bitflip), ('I', 1 - p_bitflip)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(bit_flip, ["id"])
    noise_model.add_all_qubit_quantum_error(error_gate1, ['u1', 'u2', 'u3']) # single qubit gate error is applied to all gates
    noise_model.add_all_qubit_quantum_error(error_gate1, ["id"]) # single qubit gate error is applied to identity gate = q-memory simulator
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cz"]) # two qubit gate error is applied to cx gates
        
    return noise_model

def ptheo(n, pbf):  #calculates the theoretical probability of a sigle qubit 'survivng' unaltered a falty quantum memory with pbf = probability of bitflip
    val = 0
    for i in range (0, n+1, 2):
        val += math.comb(n,i)*math.pow(pbf,i)*math.pow(1-pbf,n-i)
    return val

print("(CYCLIC) n QUBIT REPETITION CODE QEC SIMULATION\n\nProbability(bitflip) inserts an ideal noise model only on idle gate,\nsimulating a faulty quantum memory\n")
print("\nP(Depolarizing_gate_error) inserts a more realistic noise model\napplied on all 1 and 2 qubits quantum gates\n")

check = 1
while (check==1):
    n = input("\nInsert n = number of data qubits encoding the logical qubit:\n")
    if (float(n).is_integer() and float(n)>=3 and float(n)<=30 and float(n)%2!=0):
        check=2
    else:
        print("\nNon valid input: insert a odd integer number between 3 and 30 included\n")
        check=1
n = int(n)

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
#syndrome is a list of all possible syndrome measurements (in decimal and reversed because qiskit measures in reverse order)    syndrome[0:n_syndromes-1]
#positions is a list of all the corresponding positions of errors for each possible syndrome     positions[0:n_syndromes][0:n_faulty_qubits-1]

p_bf = input("NOISE MODEL\nInsert value for P(bitflip_error) = probability of flipping the qubit (X error) after the identity gate: \n")
p_bitflip = float(p_bf)
p_g = input("\nInsert value for P(depolarizing_gate_error) = probability of random Pauli error after every gate: \n")
p_gate = float(p_g)
rounds = input("\nCYCLIC QEC\nInsert number of qec rounds:\n")
rs = int(rounds)
deltat = input("\nInsert wait time units between rounds:\n")
dt = int(deltat)

#It is possible to insert a custom state for a data qubit and store it in a logical qubit, protecting the non-classical information of the qubit from noise
# theta1 = input("|psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>\nInsert value for theta [°]:\n")
# theta = math.radians(float(theta1))
# phi1 = input("\nInsert value for phi:\n")
# phi = float(phi1)

noise_model = get_noise(p_bitflip, p_gate)
aer_sim = AerSimulator(noise_model = get_noise(p_bitflip, p_gate))  #initializes noise model with p_gate and p_meas from input

#Initialization of the quantum registers: 3 data qubits, 2 ancillas, 2 classical bits
data = QuantumRegister(n, 'code_qubit')
anc = QuantumRegister(n-1, 'auxiliary_qubit')
sbit = ClassicalRegister(n-1, 'syndrome_bit')
cbit = ClassicalRegister(1, 'bypass_bit')
qc = QuantumCircuit(data, anc, sbit, cbit)

# inizialization of the data qubit in the desired state that I want to store
# |psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>
# qc.h(data[0])               #Hadamard Gate is applied to the qubit
# qc.p(theta, data[0])        #PhaseGate is applied to the qubit
# qc.h(data[0])
# qc.p((np.pi/2 + phi), data[0])

#encode the physical qubit in a n-qubit logical one
for i in range (1, n):
    qc.cx(data[0], data[i])     #control-not on the second qubit with the first qubit as control qubit

for i in range (rs):
    qc.reset(anc)   #reset ancillas to |0>
    
    for j in range (dt):
        qc.id(data).c_if(cbit, 0)   #c_if condition is always true: bypass the transpilare to compute the 'flawed' identity
    
    #entangle n-1 couples of data-qubit to n-1 ancillas to study internal correlation in the logical qubit
    for j in range (n-1):
        qc.cx(data[j], anc[j])
        qc.cx(data[j+1], anc[j])

    # ancilla_i shows if data_qubit_i = data_qubit_i+1

    #measure the ancillas to detect errors
    for j in range (n-1):
        qc.measure(anc[j], sbit[j])  #ancilla is measured and the result stored in a classical bit

    #if an error is detected in one of the qubits we apply a classically controlled not gate to correct it
    for i_syn in range (len(syndrome)):
        with qc.if_test((sbit, int(syndrome[i_syn]))):
            for j in range (len(positions[i_syn])):
                qc.x(data[positions[i_syn][j]])

print(qc)
qc.measure_all()    # measure the qubits

# run the circuit with the noise model and extract the counts
circ_n = transpile(qc, aer_sim)
result = aer_sim.run(circ_n).result()
counts = result.get_counts()

print('p_bitflip = {}\tp_gate = {}'.format(p_bitflip, p_gate))
print('\nOUTPUT:\n', counts)
print('\nOutput order: Ancilla[n-2]-...-Ancilla[0]    Data[n-1]-...-Data[0]     ByPass_Bit    Bit[n-2]-...-Bit[0]')

x = counts.keys()
oksmart = 0
for i in x:
    if ('0'*n + " 0 ") in i:
        oksmart += counts[i]

if p_bitflip == 0:
    pteo = (1-p_gate)**(rs*dt)
else:
    pteo = ptheo(rs*dt, p_bitflip)

print('\nP_success = ', oksmart/1024)
print('Probability of a single qubit surviving unaltered {} units of wait stage = {}'.format(rs*dt, pteo))

plot_histogram(counts)
plt.show()
