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


def get_noise(p_bitflip, p_phaseflip):
    bit_flip = pauli_error([('X', p_bitflip), ('I', 1 - p_bitflip)])
    phase_flip = pauli_error([('Z', p_phaseflip), ('I', 1 - p_phaseflip)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(bit_flip, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(phase_flip, ['u1', 'u2', 'u3'])
        
    return noise_model

theta1 = input("|psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>\nInsert value for theta [°]:\n")
theta = math.radians(float(theta1))
phi1 = input("\nInsert value for phi:\n")
phi = float(phi1)

p_bf = input("NOISE MODEL\nInsert value for P(bitflip_error):\n")
p_bitflip = float(p_bf)
p_pf = input("\nInsert value for P(phaseflip_error):\n")
p_phaseflip = float(p_pf)

noise_model = get_noise(p_bitflip, p_phaseflip)
aer_sim = AerSimulator(noise_model = get_noise(p_bitflip, p_phaseflip))  #initializes noise model with p_gate and p_meas from input

#Initialization of the quantum registers: 3 data qubits, 2 ancillas, 2 classical bits
data = QuantumRegister(5, 'code_qubit')
anc = QuantumRegister(4, 'auxiliary_qubit')
sbit = ClassicalRegister(4, 'syndrome_bit')
qc = QuantumCircuit(data,anc, sbit)


#encode the physical qubit in a 3-qubit logical one
qc.cx(data[0], data[1])     #control-not on the second qubit with the first qubit as control qubit
qc.cx(data[0], data[2])

# inizialization of the data qubit in the desired state that I want to store
# |psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>
qc.h(data)               #Hadamard Gate is applied to the qubit
qc.p(theta, data)        #PhaseGate is applied to the qubit
qc.h(data)
qc.p((np.pi/2 + phi), data)


#entangle 2 couples of data-qubit to 2 ancillas to study internal correlation in the logical qubit
qc.cx(data[0], anc[0])
qc.cx(data[1], anc[0])
qc.cx(data[0], anc[1])
qc.cx(data[2], anc[1])
# ancilla_0 shows if data_qubit_0 = data_qubit_1
# ancilla_1 shows if data_qubit_0 = data_qubit_2

#measure the ancillas to detect errors
qc.measure(anc[0], sbit[0])     #ancilla is measured and the result stored in a classical bit
qc.measure(anc[1], sbit[1])

#qc.draw()               #draws a schematic representation of the q-circuit

#if an error is detected in one of the qubits we apply a classically controlled not gate to correct it
qc.x(data[1]).c_if(sbit, 1)        #ATTENTION: '1' = '01' == 'sbit[1],sbit[0]'           
qc.x(data[2]).c_if(sbit, 10)       #ATTENTION: '10' == 'sbit[1],sbit[0]'        
qc.x(data[0]).c_if(sbit, 11)       #ATTENTION: '11' == 'sbit[1],sbit[0]'


qc.measure_all()    # measure the qubits

# run the circuit with the noise model and extract the counts
circ_n = transpile(qc, aer_sim)
result = aer_sim.run(circ_n).result()
counts = result.get_counts()

print('\nINPUT:\t|q0> = ({})|0> + ({})*exp(i*{})|1>'.format(np.cos(theta/2), np.sin(theta/2), phi))
print('\tp_bitflip = {}\tp_phaseflip = {}'.format(p_bitflip, p_phaseflip))
print('\nOUTPUT:\n', counts)
print('\nOutput order: Ancilla[1]-Ancilla[0]  Data[2]-Data[1]-Data[0]  Bit[1]-Bit[0]')

plot_histogram(counts)
plt.show()
