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
err = [[0] * n for i in range(nerr)]    #err[0:nerr][0:nqubit]

for i in range (nerr):
    err[i][0:(i+1)]=[1]*(i+1)   #err[i] = sequence of nerr 1s and (n-nerr) 0s

syndrome = []
positions = []

for i in range(nerr):
    for x in set(permutations(err[i])):     #evaluates all possible permutations of err[i] = all possible configurations of a n qubit string with (i+1) errors
        tempstr = ""
        templst = []
        for j in range (n-1,0,-1):
            if(list(x)[j]==list(x)[j-1]):
                tempstr = tempstr + '0'
            else:
                tempstr = tempstr + '1'
            if(list(x)[j]==1):
                templst.append(j)
        if(list(x)[0]==1):
            templst.append(0)
        positions.append(templst)
        syndrome.append(int(tempstr, base=2))
#syndrome is a list of all possible syndrome measurements (in decimal and reversed because qiskit measures in reverse order)    syndrome[0:n_syndromes]
#positions is a list of all the corresponding positions of errors for each possible syndrome     positions[0:n_syndromes][0:n_faulty_qubits]

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
    for nsyn in range (len(syndrome)):
        with qc.if_test((sbit, syndrome[nsyn])):
            for j in range (len(positions[nsyn])):
                qc.x(data[positions[nsyn][j]])

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

perr = 0
if p_bitflip == 0:
    perr = p_gate
else:
    perr = p_bitflip

pteo = ptheo(rs*dt, perr)
print('\nP_success = ', oksmart/1024)
print('Probability of a single qubit surviving unaltered {} units of wait stage = {}'.format(rs*dt, pteo))

plot_histogram(counts)
plt.show()
