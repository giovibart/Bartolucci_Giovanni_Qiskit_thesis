# Cyclic [5,1,3] Steane's code simulation with bitflip and/or depolarizing noise model.
# You can input the number of qec rounds, the wait time between rounds and the noise model parameters (error probabilities).
# The output of the program is the success probability of the code in those conditions. Also, it prints the theoretical probability of a single
# qubit surviving unaltered the same total time in a bitflip or depolarizing memory with equal error rates to the noise model.

#import qiskit
import matplotlib.pyplot as plt
import numpy as np
import math
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.tools.visualization import plot_histogram
import qiskit.quantum_info as qi
from qiskit_aer import AerSimulator

# Import from Qiskit Aer noise module
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error


def get_noise(p_bitflip, p_gate):
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

def ptheo(n, pbf):
    val = 0
    for i in range (0, n+1, 2):
        val += math.comb(n,i)*math.pow(pbf,i)*math.pow(1-pbf,n-i)
    return val

print("(CYCLIC) [5,1,3] STEANE CODE QEC SIMULATION\n\nProbability(bitflip) inserts an ideal noise model only on idle gate,\nsimulating a faulty quantum memory\n")
print("\nP(Depolarizing_gate_error) inserts a more realistic noise model\napplied on all 1 and 2 qubits quantum gates\n")
p_bf = input("NOISE MODEL\nInsert value for P(bitflip_error) = probability of flipping the qubit (X error) after the identity gate: \n")
p_bitflip = float(p_bf)
p_g = input("\nInsert value for P(depolarizing_gate_error) = probability of random Pauli error after every gate: \n")
p_gate = float(p_g)
rounds = input("\nCYCLIC QEC\nInsert number of qec rounds:\n")
rs = int(rounds)
deltat = input("\nInsert wait time units between rounds:\n")
dt = int(deltat)

# theta1 = input("|psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>\nInsert value for theta [°]:\n")
# theta = math.radians(float(theta1))
# phi1 = input("\nInsert value for phi:\n")
# phi = float(phi1)

noise_model = get_noise(p_bitflip, p_gate)
aer_sim = AerSimulator(noise_model = get_noise(p_bitflip, p_gate))  #initializes noise model with p_gate and p_meas from input

#Initialization of the quantum registers: 5 data qubits, 4 error detection ancillas, 1 measure ancilla, 4 classical bits, 1 bypass bit
data = QuantumRegister(5, 'code_qubit')
anc = QuantumRegister(4, 'auxiliary_qubit')
m_anc = QuantumRegister(1, 'measure_logic_qubit_ancilla')
sbit = ClassicalRegister(4, 'syndrome_bit')
cbit = ClassicalRegister(1, 'bypass_bit')
qc = QuantumCircuit(m_anc, data, anc, sbit, cbit)

# inizialization of the data qubit in the desired state that I want to store
# |psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>
# qc.h(data[0])               #Hadamard Gate is applied to the qubit
# qc.p(theta, data[0])        #PhaseGate is applied to the qubit
# qc.h(data[0])
# qc.p((np.pi/2 + phi), data[0])

# encode the physical qubit in a 5-qubit logical one.
# The basis state vectors are identified by the stabilizers of the [5,1,3] code
qc.z(data[0])
qc.h(data)
for i in range (1,5):
    qc.cz(data[0], data[i])
qc.h(data[0])
qc.cz(data[2], data[3])
qc.cz(data[0], data[1])
qc.cz(data[3], data[4])
qc.cz(data[1], data[2])
qc.cz(data[4], data[0])

for i in range (rs):
    qc.reset(anc)   #reset ancillas to |0>
    
    for j in range (dt):
        qc.id(data).c_if(cbit, 0)   #c_if condition is always true: bypass the transpilare to compute the 'flawed' identity
       
    # error detection using stabilizers
    qc.h(anc)

    #ancilla_0 "encodes" generator XZZXI
    qc.cx(anc[0], data[0])
    qc.cz(anc[0], data[1])
    qc.cz(anc[0], data[2])
    qc.cx(anc[0], data[3])

    #ancilla_1 encodes generator IXZZX
    qc.cx(anc[1], data[1])
    qc.cz(anc[1], data[2])
    qc.cz(anc[1], data[3])
    qc.cx(anc[1], data[4])

    #ancilla_2 encodes generator XIXZZ
    qc.cx(anc[2], data[0])
    qc.cx(anc[2], data[2])
    qc.cz(anc[2], data[3])
    qc.cz(anc[2], data[4])

    #ancilla_3 encodes generator ZXIXZ
    qc.cz(anc[3], data[0])
    qc.cx(anc[3], data[1])
    qc.cx(anc[3], data[3])
    qc.cz(anc[3], data[4])

    qc.h(anc)

    # measure the ancillas to detect errors and extrapolate info about which error and which qubit is affected
    qc.measure(anc[0], sbit[0])     #ancilla is measured and the result stored in a classical bit
    qc.measure(anc[1], sbit[1])
    qc.measure(anc[2], sbit[2])
    qc.measure(anc[3], sbit[3])

    # if an error is detected in one of the qubits we apply a classically controlled not gate to correct it
    qc.x(data[0]).c_if(sbit, 8)   #ATTENTION: '1000' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.x(data[1]).c_if(sbit, 1)      #ATTENTION: '1' = '0001' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.x(data[2]).c_if(sbit, 3)     #ATTENTION: '11' = '0011' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.x(data[3]).c_if(sbit, 6)    #ATTENTION: '110' = '0110' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.x(data[4]).c_if(sbit, 12)   #ATTENTION: '1100' = 'sbit[3],sbit[2],sbit[1],sbit[0]'

    qc.z(data[0]).c_if(sbit, 5)   #ATTENTION: '101' = '0101' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.z(data[1]).c_if(sbit, 10)  #ATTENTION: '1010' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.z(data[2]).c_if(sbit, 4)   #ATTENTION: '100' = '0100' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.z(data[3]).c_if(sbit, 9)  #ATTENTION: '1001' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.z(data[4]).c_if(sbit, 2)    #ATTENTION: '10' = '0010' == 'sbit[3],sbit[2],sbit[1],sbit[0]'

    qc.y(data[0]).c_if(sbit, 13)  #ATTENTION: '1101' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.y(data[1]).c_if(sbit, 11)  #ATTENTION: '1011' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.y(data[2]).c_if(sbit, 7)   #ATTENTION: '111' = '0111' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.y(data[3]).c_if(sbit, 15)  #ATTENTION: '1111' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.y(data[4]).c_if(sbit, 14)  #ATTENTION: '1110' == 'sbit[3],sbit[2],sbit[1],sbit[0]'

#measure the logical qubit
qc.h(m_anc)
for i in range(5):
    qc.cz(m_anc, data[i])
qc.h(m_anc)

qc.measure_all()    # measure the qubits

# run the circuit with the noise model and extract the counts
circ_n = transpile(qc, aer_sim)
result = aer_sim.run(circ_n).result()
counts = result.get_counts()

x = counts.keys()
oksmart = 0
for i in x:
    if "0 0 " in i:
        oksmart += counts[i]

if p_bitflip == 0:
    pteo = (1-p_gate)**(rs*dt)
else:
    pteo = ptheo(rs*dt, p_bitflip)

#print('\nINPUT:\t|q0> = ({})|0> + ({})*exp(i*{})|1>'.format(np.cos(theta/2), np.sin(theta/2), phi))
print('p_bitflip = {}\tp_gate = {}'.format(p_bitflip, p_gate))
print('\nOUTPUT:\n', counts)
print('\nOutput order: Ancilla[3]-Ancilla[2]-Ancilla[1]-Ancilla[0]  Data[4]-Data[3]-Data[2]-Data[1]-Data[0]-Measure_anc BypassBit Bit[3]-Bit[2]-Bit[1]-Bit[0]')

print('\nP_success = ', oksmart/1024)
print('Probability of a single qubit surviving unaltered {} units of wait stage (with p_bitflip/u = p_gate) = {}'.format(rs*dt, pteo))

plot_histogram(counts)
plt.show()

