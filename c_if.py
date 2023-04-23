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

p_bf = input("NOISE MODEL\nInsert value for P(bitflip_error):\n")
p_bitflip = float(p_bf)
p_pf = input("\nInsert value for P(phaseflip_error):\n")
p_phaseflip = float(p_pf)

noise_model = get_noise(p_bitflip, p_phaseflip)
aer_sim = AerSimulator(noise_model = get_noise(p_bitflip, p_phaseflip))  #initializes noise model with p_gate and p_meas from input

#Initialization of the quantum registers: 3 data qubits, 2 ancillas, 2 classical bits
data = QuantumRegister(3, 'code_qubit')
anc = QuantumRegister(2, 'auxiliary_qubit')
sbit = ClassicalRegister(2, 'syndrome_bit')
qc = QuantumCircuit(data, anc, sbit)

# #encode the physical qubit in a 3-qubit logical one
# qc.cx(data[0], data[1])     #control-not on the second qubit with the first qubit as control qubit
# qc.cx(data[0], data[2])

#qc.x(anc[1])
qc.x(anc[0])

#measure the ancillas to detect errors
qc.measure(anc[0], sbit[0])     #ancilla is measured and the result stored in a classical bit
qc.measure(anc[1], sbit[1])

#qc.draw()               #draws a schematic representation of the q-circuit

#if an error is detected in one of the qubits we apply a not-gate to correct it 
qc.x(data[1]).c_if(sbit, 1)                   #classically controlled gate
qc.x(data[2]).c_if(sbit, 10)
qc.x(data[0]).c_if(sbit, 11)

qc.measure_all()    # measure the qubits

# run the circuit with the noise model and extract the counts
circ_n = transpile(qc, aer_sim)
result = aer_sim.run(circ_n).result()
counts = result.get_counts()

print('\tp_bitflip = {}\tp_phaseflip = {}'.format(p_bitflip, p_phaseflip))
print('\nOUTPUT:\n', counts)

plot_histogram(counts)
plt.show()
