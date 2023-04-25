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
    noise_model.add_all_qubit_quantum_error(bit_flip, ["id"])
    noise_model.add_all_qubit_quantum_error(phase_flip, ["id"])
        
    return noise_model

#theta = np.pi
#phi = 0
p_phaseflip = 0
p_bitflip = 0.0

p_bf = np.logspace(-4,-0.5,100)
psucc = []
nocorr = []
teo = []

for p_bitflip in p_bf:
    noise_model = get_noise(p_bitflip, p_phaseflip)
    aer_sim = AerSimulator(noise_model = get_noise(p_bitflip, p_phaseflip))  #initializes noise model with p_gate and p_meas from input

    #Initialization of the quantum registers: 3 data qubits, 2 ancillas, 2 classical bits
    data = QuantumRegister(3, 'code_qubit')
    anc = QuantumRegister(2, 'auxiliary_qubit')
    sbit = ClassicalRegister(2, 'syndrome_bit')
    cbit = ClassicalRegister(1, 'simulator_bypass_bit')
    qc = QuantumCircuit(data, anc, sbit, cbit)

    #encode the physical qubit in a 3-qubit logical one
    qc.cx(data[0], data[1])     #control-not on the second qubit with the first qubit as control qubit
    qc.cx(data[0], data[2])

    # inizialization of the data qubit in the desired state that I want to store
    #  |psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>
    # qc.h(data)               #Hadamard Gate is applied to the qubit
    # qc.p(theta, data)        #PhaseGate is applied to the qubit
    # qc.h(data)
    # qc.p((np.pi/2 + phi), data)

    qc.id(data).c_if(cbit, 0)

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

    #if an error is detected in one of the qubits we apply a classically controlled not gate to correct it
    qc.x(data[1]).c_if(sbit, 1)        #ATTENTION: '1' = '01' == 'sbit[1],sbit[0]'           
    qc.x(data[2]).c_if(sbit, 10)       #ATTENTION: '10' == 'sbit[1],sbit[0]'        
    qc.x(data[0]).c_if(sbit, 11)       #ATTENTION: '11' == 'sbit[1],sbit[0]'


    qc.measure_all()    # measure the qubits

    # run the circuit with the noise model and extract the counts
    circ_n = transpile(qc, aer_sim)
    result = aer_sim.run(circ_n).result()
    counts = result.get_counts()

    x = counts.keys()
    okcounts = 0
    oknocorr = 0
    for i in x:
        if i == "00000 0 00":
            okcounts += counts[i]
            oknocorr += counts[i]
        elif i == "01000 0 01":
            okcounts += counts[i]
        elif i == "10000 0 10":
            okcounts += counts[i]
        elif i == "11000 0 11":
            okcounts += counts[i]

    psucc.append(float(okcounts/1024))
    nocorr.append(float(oknocorr/1024))
    teo.append(1 - p_bitflip)

P_success = np.array(psucc)
P_nocorr = np.array(nocorr)
P_theo = np.array(teo)
x_y = np.array([p_bf, P_success])

np.savetxt("Psuccess_vs_Pbf.txt", x_y)

plt.plot(p_bf, P_success, marker='.', color='red')
plt.plot(p_bf, P_nocorr, marker='.', color='blue')
plt.plot(p_bf, P_theo, marker='.', color='green')
plt.grid()
plt.xlabel("P(bitflip)", fontsize=8, fontname='Sans')
plt.ylabel("P(success) = (correct final state counts)/(total counts)", fontsize=8, fontname='Sans')
plt.title("Probability of getting target-state after QEC as a function of bitflip probability", fontsize=10, fontname='Sans', fontweight='bold')
plt.suptitle("QEC: 3-qubit code [3,1]", fontsize=10, fontname='Sans', fontweight='bold')
plt.xscale('log')
plt.legend(["with QEC", "without correction", "P(success, 1 qubit, no QEC)"], fontsize=10, loc='lower left')
#plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks([10**(-4), 10**(-3), 10**(-2), 0.1, 0.3])
plt.xlim([1e-04, 0.33])
#plt.ylim([0.7,1])
plt.savefig("Psuccess_vs_Pbf.png", dpi=1000)
plt.show()