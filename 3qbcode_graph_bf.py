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


def get_noise(p_bitflip):
    bit_flip = pauli_error([('X', p_bitflip), ('I', 1 - p_bitflip)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(bit_flip, ["id"])
        
    return noise_model

def ptheo(n, pbf):
    val = 0
    for i in range (0, n+1, 2):
        val += math.comb(n,i)*math.pow(pbf,i)*math.pow(1-pbf,n-i)
    return val


p_bitflip = 0.0

p_bf = np.logspace(-4,-0.5,100)
pteo_32u = []
pfail_teo = []

zoomb = 10**(-2.5) #zoom boundary
posiz = 0
for i in range(100):
    if p_bf[i] <= zoomb:
        posiz = i
    else:
        i=101

p_bf_fail = p_bf[0:posiz+1]

round = np.array([1,2,4,8,16,32])
dt = np.array([32,16,8,4,2,1])
psucc = np.zeros((6,100), dtype='float64')
pfail = np.zeros((6,posiz+1), dtype='float64')
col = -1

#delta_t = 2u
for riga in range(6):
    col = -1
    for p_bitflip in p_bf:
        col = col+1
        noise_model = get_noise(p_bitflip)
        aer_sim = AerSimulator(noise_model = get_noise(p_bitflip))  #initializes noise model with p_gate and p_meas from input

        #Initialization of the quantum registers: 3 data qubits, 2 ancillas, 2 classical bits
        data = QuantumRegister(3, 'code_qubit')
        anc = QuantumRegister(2, 'auxiliary_qubit')
        sbit = ClassicalRegister(2, 'syndrome_bit')
        cbit = ClassicalRegister(1, 'simulator_bypass_bit')
        qc = QuantumCircuit(data, anc, sbit, cbit)

        #encode the physical qubit in a 3-qubit logical one
        qc.cx(data[0], data[1])     #control-not on the second qubit with the first qubit as control qubit
        qc.cx(data[0], data[2])

        for i in range (round[riga]):
            qc.reset(anc)
            for j in range (dt[riga]):
                qc.id(data).c_if(cbit, 0) #bypass the transpiler to compute identity
            
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
        for it in x:
            if it == "00000 0 00":
                okcounts += counts[it]
            elif it == "01000 0 01":
                okcounts += counts[it]
            elif it == "10000 0 10":
                okcounts += counts[it]
            elif it == "11000 0 11":
                okcounts += counts[it]

        psucc[riga][col] = float(okcounts/1024)
        if (riga==0):
            pteo_32u.append(ptheo(32, p_bitflip))
            if (p_bitflip <= zoomb):
                pfail_teo.append(float(1-ptheo(32, p_bitflip)))

        if (p_bitflip <= zoomb):
            pfail[riga][col] = float(1-okcounts/1024)

Pteo_32u = np.array(pteo_32u)
Pfail_teo = np.array(pfail_teo)

P_cu = psucc[0]
P_c2u = psucc[1]
P_c4u = psucc[2]
P_c8u = psucc[3]
P_c16u = psucc[4]
P_c32u = psucc[5]

Pfail_u = pfail[0]
Pfail_2u = pfail[1]
Pfail_4u = pfail[2]
Pfail_8u = pfail[3]
Pfail_16u = pfail[4]
Pfail_32u = pfail[5]


plt.plot(p_bf, P_cu, marker='.', color='black')
plt.plot(p_bf, P_c2u, marker='.', color='red')
plt.plot(p_bf, P_c4u, marker='.', color='blue')
plt.plot(p_bf, P_c8u, marker='.', color='green')
plt.plot(p_bf, P_c16u, marker='.', color='pink')
plt.plot(p_bf, P_c32u, marker='.', color='brown')
plt.plot(p_bf, Pteo_32u, marker='+', color='purple')

plt.grid()
plt.xlabel("P(bitflip) [logscale]", fontsize=15, fontname='Sans')
plt.ylabel("P(success) = (correct final state counts)/(total counts)", fontsize=15, fontname='Sans')
plt.title("Probability of getting target-state after QEC as a function of bitflip probability", fontsize=20, fontname='Sans', fontweight='bold')
plt.suptitle("QEC: cyclic 3-qubit code [3,1]", fontsize=20, fontname='Sans', fontweight='bold')
plt.xscale('log')
plt.legend(["with QEC rounds=1 dt=32u", "with QEC rounds=2 dt=16u", "with QEC r=4 dt=8u", "with QEC r=8 dt=4u", 
                "with QEC r=16 dt=2u","with QEC r=32 dt=1u", "1 qubit, no correction for dt=32u"], fontsize=14, loc='lower left')
#plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks([10**(-4), 10**(-3), 10**(-2), 0.1, 0.3])
plt.xlim([1e-04, 0.33])
#plt.ylim([0.7,1])
plt.savefig("cyc3_Psucc_vs_Pbf.png", dpi=1000)
plt.show()

plt.plot(p_bf_fail, Pfail_u, marker='.', color='black')
plt.plot(p_bf_fail, Pfail_2u, marker='.', color='red')
plt.plot(p_bf_fail, Pfail_4u, marker='.', color='blue')
plt.plot(p_bf_fail, Pfail_8u, marker='.', color='green')
plt.plot(p_bf_fail, Pfail_16u, marker='.', color='pink')
plt.plot(p_bf_fail, Pfail_32u, marker='.', color='brown')
#plt.plot(p_bf_fail, Pfail_teo, marker='+', color='purple')

plt.grid()
plt.xlabel("P(bitflip) [logscale]", fontsize=15, fontname='Sans')
plt.ylabel("P(failure) = 1 - P(success) [logscale]", fontsize=15, fontname='Sans')
plt.title("Probability of getting wrong state after QEC as a function of bitflip probability", fontsize=20, fontname='Sans', fontweight='bold')
plt.suptitle("QEC: cyclic 3-qubit code [3,1]", fontsize=20, fontname='Sans', fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.legend(["with QEC rounds=1 dt=32u", "with QEC rounds=2 dt=16u", "with QEC r=4 dt=8u", "with QEC r=8 dt=4u", 
                "with QEC r=16 dt=2u","with QEC r=32 dt=1u"], fontsize=14, loc='upper left')    #, "1 qubit, no correction for dt=32u"

plt.xticks([10**(-4), 10**(-3), zoomb])
plt.xlim([1e-04, zoomb])
plt.savefig("cyc3_Pfail_vs_Pbf.png", dpi=1000)
plt.show()