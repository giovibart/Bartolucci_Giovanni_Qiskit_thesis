#import qiskit
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.tools.visualization import plot_histogram
import qiskit.quantum_info as qi
from qiskit_aer import AerSimulator

# Import from Qiskit Aer noise module
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error


def get_noise(p_meas,p_gate):    
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ['u1', 'u2', 'u3']) # single qubit gate error is applied to all gates
    noise_model.add_all_qubit_quantum_error(error_gate1, ["id"]) # single qubit gate error is applied to identity gate = q-memory simulator
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cz"]) # two qubit gate error is applied to cz gates    
    return noise_model

noise_model = get_noise(0.0, 0.01)
aer_sim = AerSimulator(noise_model = get_noise(0.0, 0.01))  #initializes noise model with p_gate = p_meas = 1%

# initialization of a Quantum Circuit with three qubits in the 0 state (|000>)

data = QuantumRegister(3, 'code_qubit')
cbit = ClassicalRegister(1, 'simulator_bypass_bit')
qc0 = QuantumCircuit(data, cbit)

#qc0.id(data).c_if(cbit, 0)
# qc0.cz(data[0], data[1])
# qc0.cz(data[0], data[1]).c_if(cbit, 0)
#qc0.x(data)
#qc0.x(data).c_if(cbit, 0)
#qc0.p(np.pi, data)
#qc0.p(np.pi, data).c_if(cbit, 0)
# qc0.z(data[0])
# qc0.z(data[0]).c_if(cbit, 0)
qc0.h(data[0])
qc0.h(data[0]).c_if(cbit, 0)        
qc0.measure_all()

# run the circuit with the noise model and extract the counts
circ_n = transpile(qc0, aer_sim)
result = aer_sim.run(circ_n).result()
counts = result.get_counts()

print('\nINPUT |000>, p_gate = p_meas = 1%, OUTPUT:\n', counts)

plot_histogram(counts)
plt.show()
