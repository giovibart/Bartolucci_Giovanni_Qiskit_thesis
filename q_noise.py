#import qiskit
import matplotlib.pyplot as plt
import numpy as np
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.tools.visualization import plot_histogram
import qiskit.quantum_info as qi
from qiskit_aer import AerSimulator

aer_sim = Aer.get_backend('aer_simulator')

def get_noise(p_meas,p_gate):
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
        
    return noise_model

noise_model = get_noise(0.01,0.01)  #initializes noise model with p_gate = p_meas = 1%

# initialization of a Quantum Circuit with three qubits in the 0 state (|000>)

qc0 = QuantumCircuit(3) 
qc0.measure_all() # measure the qubits

# run the circuit with the noise model and extract the counts
qobj = assemble(qc0)
counts = aer_sim.run(qobj, noise_model=noise_model).result().get_counts()

print('\nINPUT |000>, p_gate = p_meas = 1%, OUTPUT:\n', counts)

#plot_histogram(counts)

qc1 = QuantumCircuit(3) # initialize circuit with three qubits in the 0 state
qc1.x([0,1,2]) # flip each 0 to 1

qc1.measure_all() # measure the qubits

# run the circuit with th noise model and extract the counts
qobj = assemble(qc1)
counts2 = aer_sim.run(qobj, noise_model=noise_model).result().get_counts()
print('\nINPUT |111>, p_gate = p_meas = 1%, OUTPUT:\n', counts2)

noise_model = get_noise(0.5,0.0)    # p_meas = 0.5   p_gate = 0
qobj = assemble(qc1)
counts3 = aer_sim.run(qobj, noise_model=noise_model).result().get_counts()
print('\nINPUT |111>, p_gate = 0% p_meas = 50%, OUTPUT:\n', counts3)

#initialization of a Quantum circuit with 2 code qubits, 1 auxiliary qubit (ancilla) all in the 0 state
# + one classical bit to store the ancilla measurement

cq = QuantumRegister(2, 'code_qubit')
lq = QuantumRegister(1, 'auxiliary_qubit')
sb = ClassicalRegister(1, 'syndrome_bit')
qc = QuantumCircuit(cq, lq, sb)
qc.cx(cq[0], lq[0])     #control-not on the ancilla with the first code-qubit as control qubit
qc.cx(cq[1], lq[0])     #control-not on the ancilla with the second code-qubit as control qubit
qc.measure(lq, sb)      #ancilla is measured and the result stored in the classical bit
#qc.draw()               #draws a schematic representation of the q-circuit

print('\nCASE 0: all code-qubits are initialized as |0> (ancilla si always initialized as |0>)\n')
qc_init = QuantumCircuit(cq)
#qc.compose(qc_init).draw()
qobj = assemble(qc.compose(qc_init, front=True))
counts = aer_sim.run(qobj).result().get_counts()
print('Results of CASE 0:\n',counts)

print('\nCASE 1: all code-qubits are initialized as |1>\n')
qc_init = QuantumCircuit(cq)
qc_init.x(cq)
#qc.compose(qc_init, front=True).draw()
qobj = assemble(qc.compose(qc_init, front=True))
counts = aer_sim.run(qobj).result().get_counts()
print('Results of CASE 1:\n',counts)

print('\nCASE 2: code qubits are initialized as (1/sqrt(2))*(|00> + |11>)\n')
qc_init = QuantumCircuit(cq)
qc_init.h(cq[0])
qc_init.cx(cq[0], cq[1])
#qc.compose(qc_init, front=True).draw()
qobj = assemble(qc.compose(qc_init, front=True))
counts = aer_sim.run(qobj).result().get_counts()
print('Results of CASE 2:\n',counts)

print('\nCASE 3: code qubits are initialized as (1/sqrt(2))*(|01> + |10>)\n')
qc_init = QuantumCircuit(cq)
qc_init.h(cq[0])
qc_init.cx(cq[0], cq[1])
qc_init.x(cq[0])
#qc.compose(qc_init, front=True).draw()
qobj = assemble(qc.compose(qc_init, front=True))
counts = aer_sim.run(qobj).result().get_counts()
print('Results of CASE 3:\n',counts)

