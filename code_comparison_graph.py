#Graph producing program for 3qubit_repetition vs 5qubit_repetition vs [5,1,3]Steane codes comparison
#with depolarizing noise model on all gates and fixed probability of bitflip error on identity gate
#Probability of depolarizing gate error is sweeped between 10^(-4) and 0.032 in logscale and resulting P_success of each code is plotted
#The produced graph is helpful to identify the range of gate errors where the codes are effective in correcting bitflip errors happening during the waiting time
#In fact, usually the most errors happen because data qubits decohere while they idle during measurement and reset of ancillas between rounds of error correction
#Furthermore the various codes performance and range are this way directly comparable

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
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)
    bit_flip = pauli_error([('X', p_bitflip), ('I', 1 - p_bitflip)])

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(bit_flip, ["id"])
    noise_model.add_all_qubit_quantum_error(error_gate1, ['u1', 'u2', 'u3']) # single qubit gate error is applied to all gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cz"]) # two qubit gate error is applied to cx gates
        
    return noise_model

def ptheo(n, pbf):
    val = 0
    for i in range (0, n+1, 2):
        val += math.comb(n,i)*math.pow(pbf,i)*math.pow(1-pbf,n-i)
    return val

p_bitflip = (4.4e-2)*1.2    #error rate during waiting time. Source: Exponential suppression of bit or phase errors with cyclic error correction
                            # +20% (experimentally measured)
p_gate = 0.0
p_g = np.logspace(-4,-1.7,100)
psucc_3 = []
teo = []

for p_gate in p_g:
    noise_model = get_noise(p_bitflip, p_gate)
    aer_sim = AerSimulator(noise_model = get_noise(p_bitflip, p_gate))  #initializes noise model with p_gate and p_meas from input

    #Initialization of the quantum registers: 3 data qubits, 2 ancillas, 2 classical bits
    data = QuantumRegister(3, 'code_qubit')
    anc = QuantumRegister(2, 'auxiliary_qubit')
    sbit = ClassicalRegister(2, 'syndrome_bit')
    cbit = ClassicalRegister(1, 'simulator_bypass_bit')
    qc = QuantumCircuit(data, anc, sbit, cbit)

    qc.reset(anc)
    #encode the physical qubit in a 3-qubit logical one
    qc.cx(data[0], data[1])     #control-not on the second qubit with the first qubit as control qubit
    qc.cx(data[0], data[2])

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
    qc.x(data[2]).c_if(sbit, 2)       #ATTENTION: '10' == 'sbit[1],sbit[0]'        
    qc.x(data[0]).c_if(sbit, 3)       #ATTENTION: '11' == 'sbit[1],sbit[0]'


    qc.measure_all()    # measure the qubits

    # run the circuit with the noise model and extract the counts
    circ_n = transpile(qc, aer_sim)
    result = aer_sim.run(circ_n).result()
    counts = result.get_counts()

    x = counts.keys()
    okcounts = 0
    for i in x:
        if i == "00000 0 00":
            okcounts += counts[i]
        elif i == "01000 0 01":
            okcounts += counts[i]
        elif i == "10000 0 10":
            okcounts += counts[i]
        elif i == "11000 0 11":
            okcounts += counts[i]

    psucc_3.append(float(okcounts/1024))
    teo.append(1 - p_bitflip)

P_success_3 = np.array(psucc_3)
P_theo = np.array(teo)

psucc_5 = []
p_gate = 0.0
for p_gate in p_g:
    noise_model = get_noise(p_bitflip, p_gate)
    aer_sim = AerSimulator(noise_model = get_noise(p_bitflip, p_gate))  #initializes noise model with p_gate and p_meas from input

    #Initialization of the quantum registers: 5 data qubits, 4 ancillas, 4 classical bits, 1 simulator bypass bit
    data = QuantumRegister(5, 'code_qubit')
    anc = QuantumRegister(4, 'auxiliary_qubit')
    sbit = ClassicalRegister(4, 'syndrome_bit')
    cbit = ClassicalRegister(1, 'simulator_bypass_bit')
    qc = QuantumCircuit(data, anc, sbit, cbit)

    qc.reset(anc)

    #encode the physical qubit in a 5-qubit logical one
    qc.cx(data[0], data[1])     #control-not on the second qubit with the first qubit as control qubit
    qc.cx(data[0], data[2])
    qc.cx(data[0], data[3])
    qc.cx(data[0], data[4])

    qc.id(data).c_if(cbit, 0)

    #entangle 4 couples of data-qubit to 4 ancillas to study internal correlation in the logical qubit
    qc.cx(data[0], anc[0])
    qc.cx(data[1], anc[0])
    qc.cx(data[1], anc[1])
    qc.cx(data[2], anc[1])
    qc.cx(data[2], anc[2])
    qc.cx(data[3], anc[2])
    qc.cx(data[3], anc[3])
    qc.cx(data[4], anc[3])
    # ancilla_0 shows if data_qubit_0 = data_qubit_1
    # ancilla_1 shows if data_qubit_1 = data_qubit_2
    # ancilla_2 shows if data_qubit_2 = data_qubit_3
    # ancilla_3 shows if data_qubit_3 = data_qubit_4

    #measure the ancillas to detect errors
    qc.measure(anc[0], sbit[0])     #ancilla is measured and the result stored in a classical bit
    qc.measure(anc[1], sbit[1])
    qc.measure(anc[2], sbit[2])
    qc.measure(anc[3], sbit[3])

    #if an error is detected in one of the qubits we apply a classically controlled not gate to correct it
    #single error events
    qc.x(data[0]).c_if(sbit, 1)        #ATTENTION: '1' = '0001' == 'sbit[3],sbit[2],sbit[1],sbit[0]'           
    qc.x(data[1]).c_if(sbit, 3)       #ATTENTION: '11' = '0011' == 'sbit[3],sbit[2],sbit[1],sbit[0]'        
    qc.x(data[2]).c_if(sbit, 6)       #ATTENTION: '110' = '0110 == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.x(data[3]).c_if(sbit, 12)       #ATTENTION: '1100' == 'sbit[3],sbit[2],sbit[1],sbit[0]'
    qc.x(data[4]).c_if(sbit, 8)       #ATTENTION: '1000' == 'sbit[3],sbit[2],sbit[1],sbit[0]'

    #2 error events
    qc.x(data[0]).c_if(sbit, 2)
    qc.x(data[1]).c_if(sbit, 2)

    qc.x(data[0]).c_if(sbit, 7)
    qc.x(data[2]).c_if(sbit, 7)

    qc.x(data[0]).c_if(sbit, 13)
    qc.x(data[3]).c_if(sbit, 13)

    qc.x(data[0]).c_if(sbit, 9)
    qc.x(data[4]).c_if(sbit, 9)

    qc.x(data[1]).c_if(sbit, 5)
    qc.x(data[2]).c_if(sbit, 5)

    qc.x(data[1]).c_if(sbit, 15)
    qc.x(data[3]).c_if(sbit, 15)

    qc.x(data[1]).c_if(sbit, 11)
    qc.x(data[4]).c_if(sbit, 11)

    qc.x(data[2]).c_if(sbit, 10)
    qc.x(data[3]).c_if(sbit, 10)

    qc.x(data[2]).c_if(sbit, 14)
    qc.x(data[4]).c_if(sbit, 14)

    qc.x(data[3]).c_if(sbit, 4)
    qc.x(data[4]).c_if(sbit, 4)


    qc.measure_all()    # measure the qubits

    # run the circuit with the noise model and extract the counts
    circ_n = transpile(qc, aer_sim)
    result = aer_sim.run(circ_n).result()
    counts = result.get_counts()

    x = counts.keys()
    okcounts = 0
    for i in x:
        if "00000 0 " in i:
            okcounts += counts[i]

    psucc_5.append(float(okcounts/1024))

P_success_5 = np.array(psucc_5)

psucc_ste = []
p_gate = 0.0
for p_gate in p_g:
    noise_model = get_noise(p_bitflip, p_gate)
    aer_sim = AerSimulator(noise_model = get_noise(p_bitflip, p_gate))  #initializes noise model with p_gate and p_meas from input

    #Initialization of the quantum registers: 5 data qubits, 4 ancillas, 4 classical bits, 1 simulator bypass bit
    data = QuantumRegister(5, 'code_qubit')
    anc = QuantumRegister(4, 'auxiliary_qubit')
    m_anc = QuantumRegister(1, 'measure_logic_qubit_ancilla')
    sbit = ClassicalRegister(4, 'syndrome_bit')
    cbit = ClassicalRegister(1, 'bypass_bit')
    qc = QuantumCircuit(m_anc, data, anc, sbit, cbit)

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

    qc.id(data).c_if(cbit, 0)


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
    okcounts = 0
    for i in x:
        if "0 0 " in i:
            okcounts += counts[i]

    psucc_ste.append(float(okcounts/1024))

P_success_ste = np.array(psucc_ste)

plt.plot(p_g, P_success_3, marker='.', color='red')
plt.plot(p_g, P_success_5, marker='.', color='blue')
plt.plot(p_g, P_success_ste, marker='.', color='green')
plt.plot(p_g, P_theo, marker='+', color='purple')
plt.grid()
plt.xlabel("P(depolarizing gate error)", fontsize=15, fontname='Sans')
plt.ylabel("P(success) = (correct final state counts)/(total counts)", fontsize=15, fontname='Sans')
plt.title("Probability of getting target-state after QEC as a function of depolarizing gate error probability at fixed P(bitflip) on Id", fontsize=18, fontname='Sans', fontweight='bold')
plt.suptitle("QEC realistic simulation: 3-qubit_repetition vs 5-qubit_repetition vs [5,1,3]-Steane codes", fontsize=20, fontname='Sans', fontweight='bold')
plt.xscale('log')
plt.legend(["3 qubit repetition code", "5 qubit repetition code", "[5,1,3]-Steane code", "P(success, 1 qubit, no QEC)"], fontsize=15, loc='lower left')
#plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks([10**(-4), 10**(-3), 10**(-2)])
plt.xlim([1e-04, 2e-02])
#plt.ylim([0.7,1])
plt.savefig("3qb_Psuccess_vs_Pgate@fixed-Pbf.png", dpi=1000)
plt.show()