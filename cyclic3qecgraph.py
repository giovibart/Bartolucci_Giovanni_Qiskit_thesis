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

def ptheo(n, pbf):
    val = 0
    for i in range (0, n+1, 2):
        val += math.comb(n,i)*math.pow(pbf,i)*math.pow(1-pbf,n-i)
    return val


p_phaseflip = 0
p_bitflip = 0.0

p_bf = np.logspace(-4,-0.5,100)
p_c2u = []
pteo_2ux5 = []
p_c4u = []
pteo_4ux5 = []
p_c8u = []
pteo_8ux5 = []
p_c16u = []
pteo_16ux5 = []
p_c32u = []
pteo_32ux5 = []


#delta_t = 2u

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

    for i in range (5):
        qc.reset(anc)
    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        #repeat n times to set waiting time to 2n units of time

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
    for i in x:
        if i == "00000 0 00":
            okcounts += counts[i]
        elif i == "01000 0 01":
            okcounts += counts[i]
        elif i == "10000 0 10":
            okcounts += counts[i]
        elif i == "11000 0 11":
            okcounts += counts[i]

    p_c2u.append(float(okcounts/1024))
    pteo_2ux5.append(ptheo(10, p_bitflip))

P_c2u = np.array(p_c2u)
Pteo_2ux5 = np.array(pteo_2ux5)

#delta_t = 4u

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

    for i in range (5):
        qc.reset(anc)
    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity

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
    for i in x:
        if i == "00000 0 00":
            okcounts += counts[i]
        elif i == "01000 0 01":
            okcounts += counts[i]
        elif i == "10000 0 10":
            okcounts += counts[i]
        elif i == "11000 0 11":
            okcounts += counts[i]

    p_c4u.append(float(okcounts/1024))
    pteo_4ux5.append(ptheo(20, p_bitflip))

P_c4u = np.array(p_c4u)
Pteo_4ux5 = np.array(pteo_4ux5)

#delta_t = 8u

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

    for i in range (5):
        qc.reset(anc)
    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity

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
    for i in x:
        if i == "00000 0 00":
            okcounts += counts[i]
        elif i == "01000 0 01":
            okcounts += counts[i]
        elif i == "10000 0 10":
            okcounts += counts[i]
        elif i == "11000 0 11":
            okcounts += counts[i]

    p_c8u.append(float(okcounts/1024))
    pteo_8ux5.append(ptheo(40, p_bitflip))

P_c8u = np.array(p_c8u)
Pteo_8ux5 = np.array(pteo_8ux5)


#delta_t = 16u

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

    for i in range (5):
        qc.reset(anc)
    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity

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
    for i in x:
        if i == "00000 0 00":
            okcounts += counts[i]
        elif i == "01000 0 01":
            okcounts += counts[i]
        elif i == "10000 0 10":
            okcounts += counts[i]
        elif i == "11000 0 11":
            okcounts += counts[i]

    p_c16u.append(float(okcounts/1024))
    pteo_16ux5.append(ptheo(80, p_bitflip))

P_c16u = np.array(p_c16u)
Pteo_16ux5 = np.array(pteo_16ux5)


#delta_t = 32u

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

    for i in range (5):
        qc.reset(anc)
    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity    
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity
        qc.id(data)
        qc.id(data).c_if(cbit, 0)    #bypass the transpiler to compute identity

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
    for i in x:
        if i == "00000 0 00":
            okcounts += counts[i]
        elif i == "01000 0 01":
            okcounts += counts[i]
        elif i == "10000 0 10":
            okcounts += counts[i]
        elif i == "11000 0 11":
            okcounts += counts[i]

    p_c32u.append(float(okcounts/1024))
    pteo_32ux5.append(ptheo(160, p_bitflip))

P_c32u = np.array(p_c32u)
Pteo_32ux5 = np.array(pteo_32ux5)




plt.plot(p_bf, P_c2u, marker='.', color='red')
plt.plot(p_bf, Pteo_2ux5, marker='.', color='orange')
plt.plot(p_bf, P_c4u, marker='.', color='blue')
plt.plot(p_bf, Pteo_4ux5, marker='.', color='cyan')
plt.plot(p_bf, P_c8u, marker='.', color='green')
plt.plot(p_bf, Pteo_8ux5, marker='.', color='olive')
plt.plot(p_bf, P_c16u, marker='.', color='purple')
plt.plot(p_bf, Pteo_16ux5, marker='.', color='pink')
plt.plot(p_bf, P_c32u, marker='.', color='brown')
plt.plot(p_bf, Pteo_32ux5, marker='.', color='gray')

plt.grid()
plt.xlabel("P(bitflip)", fontsize=8, fontname='Sans')
plt.ylabel("P(success) = (correct final state counts)/(total counts)", fontsize=8, fontname='Sans')
plt.title("Probability of getting target-state after QEC as a function of bitflip probability", fontsize=10, fontname='Sans', fontweight='bold')
plt.suptitle("QEC: (5)cyclic 3-qubit code [3,1]", fontsize=10, fontname='Sans', fontweight='bold')
plt.xscale('log')
plt.legend(["with QEC dt=2u", "no corr dt=2u*5=10u", "with QEC dt=4u", "no corr dt=4u*5=20u",
            "with QEC dt=8u", "no corr dt=8u*5=40u","with QEC dt=16u", "no corr dt=16u*5=80u",
            "with QEC dt=32u", "no corr dt=32u*5=160u"], fontsize=10, loc='lower left')
#plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks([10**(-4), 10**(-3), 10**(-2), 0.1, 0.3])
plt.xlim([1e-04, 0.33])
#plt.ylim([0.7,1])
plt.savefig("Psuccess_vs_Pbf.png", dpi=1000)
plt.show()