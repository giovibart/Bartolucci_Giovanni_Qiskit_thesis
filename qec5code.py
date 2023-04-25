#import qiskit
import matplotlib.pyplot as plt
import numpy as np
import math
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.tools.visualization import plot_histogram
import qiskit.quantum_info as qi
from qiskit.quantum_info import DensityMatrix, StabilizerState
from qiskit_aer import AerSimulator

# Import from Qiskit Aer noise module
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error


def get_noise(p_bitflip, p_phaseflip):
    bit_flip = pauli_error([('X', p_bitflip), ('I', 1 - p_bitflip)])
    phase_flip = pauli_error([('Z', p_phaseflip), ('I', 1 - p_phaseflip)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(bit_flip, ["id"])   #['u1', 'u2', 'u3']
    noise_model.add_all_qubit_quantum_error(phase_flip, ['u1', 'u2', 'u3'])
        
    return noise_model

# theta1 = input("|psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>\nInsert value for theta [°]:\n")
# theta = math.radians(float(theta1))
# phi1 = input("\nInsert value for phi:\n")
# phi = float(phi1)

p_bf = input("NOISE MODEL\nInsert value for P(bitflip_error):\n")
p_bitflip = float(p_bf)
p_pf = input("\nInsert value for P(phaseflip_error):\n")
p_phaseflip = float(p_pf)

noise_model = get_noise(p_bitflip, p_phaseflip)
aer_sim = AerSimulator(noise_model = get_noise(p_bitflip, p_phaseflip))  #initializes noise model with p_gate and p_meas from input

#Initialization of the quantum registers: 3 data qubits, 2 ancillas, 2 classical bits
data = QuantumRegister(5, 'code_qubit')
anc = QuantumRegister(4, 'auxiliary_qubit')
m_anc = QuantumRegister(1, 'measure_logic_qubit_ancilla')
sbit = ClassicalRegister(4, 'syndrome_bit')
cbit = ClassicalRegister(1, 'bypass_bit')
qc = QuantumCircuit(m_anc, data, anc, sbit, cbit)

rho = DensityMatrix(qc)

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

# alternatively we can initialize the state measuring the stabilizers 

stabstate = StabilizerState(qc)
rho = rho.evolve(stabstate)

# apply error channel
qc.id(data).c_if(cbit, 0)
#qc.id(data).c_if(cbit, 0)
#repeat n times

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

#qc.draw()      #draws a schematic representation of the q-circuit

# if an error is detected in one of the qubits we apply a classically controlled not gate to correct it

qc.x(data[0]).c_if(sbit, 1000)   #ATTENTION: '1000' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.x(data[1]).c_if(sbit, 1)      #ATTENTION: '1' = '0001' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.x(data[2]).c_if(sbit, 11)     #ATTENTION: '11' = '0011' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.x(data[3]).c_if(sbit, 110)    #ATTENTION: '110' = '0110' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.x(data[4]).c_if(sbit, 1100)   #ATTENTION: '1100' = 'sbit[3],sbit[2],sbit[1],sbit[0]'

qc.z(data[0]).c_if(sbit, 101)   #ATTENTION: '101' = '0101' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.z(data[1]).c_if(sbit, 1010)  #ATTENTION: '1010' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.z(data[2]).c_if(sbit, 100)   #ATTENTION: '100' = '0100' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.z(data[3]).c_if(sbit, 1001)  #ATTENTION: '1001' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.z(data[4]).c_if(sbit, 10)    #ATTENTION: '10' = '0010' == 'sbit[3],sbit[2],sbit[1],sbit[0]'

qc.y(data[0]).c_if(sbit, 1101)  #ATTENTION: '1101' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.y(data[1]).c_if(sbit, 1011)  #ATTENTION: '1011' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.y(data[2]).c_if(sbit, 111)   #ATTENTION: '111' = '0111' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.y(data[3]).c_if(sbit, 1111)  #ATTENTION: '1111' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
qc.y(data[4]).c_if(sbit, 1110)  #ATTENTION: '1110' == 'sbit[3],sbit[2],sbit[1],sbit[0]'


# measure the logical qubit
qc.h(m_anc)
for i in range(5):
    qc.cz(m_anc, data[i])
qc.h(m_anc)

# decode the logical qubit
qc.cz(data[4], data[0])
qc.cz(data[1], data[2])
qc.cz(data[3], data[4])
qc.cz(data[0], data[1])
qc.cz(data[2], data[3])
qc.h(data[0])
for i in range (4,0,-1):
    qc.cz(data[0], data[i])
qc.h(data)
qc.z(data[0])

# measure all the physical qubits
qc.measure_all()    

# run the circuit with the noise model and extract the counts
circ_n = transpile(qc, aer_sim)
result = aer_sim.run(circ_n).result()
counts = result.get_counts()

x = counts.keys()

okcounts = 0
for i in x:
    if i == "1000000000 0 1000":
        okcounts += counts[i]
    elif i == "0001000000 0 0001":
        okcounts += counts[i]
    elif i == "0011000000 0 0011":
        okcounts += counts[i]
    elif i == "0110000000 0 0110":
        okcounts += counts[i]
    elif i == "1100000000 0 1100":
        okcounts += counts[i]
    elif i == "0101000000 0 0101":
        okcounts += counts[i]
    elif i == "1010000000 0 1010":
        okcounts += counts[i]
    elif i == "0100000000 0 0100":
        okcounts += counts[i]
    elif i == "1001000000 0 1001":
        okcounts += counts[i]
    elif i == "0010000000 0 0010":
        okcounts += counts[i]
    elif i == "1101000000 0 1101":
        okcounts += counts[i]
    elif i == "1011000000 0 1011":
        okcounts += counts[i]
    elif i == "0111000000 0 0111":
        okcounts += counts[i]
    elif i == "1111000000 0 1111":
        okcounts += counts[i]
    elif i == "1110000000 0 1110":
        okcounts += counts[i]
    elif i == "0000000000 0 0000":
        okcounts += counts[i]

oksmart = 0
for i in x:
    if "0 0 " in i:
        oksmart += counts[i]



#print('\nINPUT:\t|q0> = ({})|0> + ({})*exp(i*{})|1>'.format(np.cos(theta/2), np.sin(theta/2), phi))
print('\tp_bitflip = {}\tp_phaseflip = {}'.format(p_bitflip, p_phaseflip))
print('\nOUTPUT:\n', counts)
print('\nOutput order: Ancilla[3]-Ancilla[2]-Ancilla[1]-Ancilla[0]  Data[4]-Data[3]-Data[2]-Data[1]-Data[0] BypassBit Bit[3]-Bit[2]-Bit[1]-Bit[0]')

print('\n\nP_success = ', okcounts/1024)
print('\nP_succ_smart = ', oksmart/1024)
plot_histogram(counts)
plt.show()
