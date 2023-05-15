#Graph producing program for [5,1,3] Steane QEC code with bitflip noise only on idetity gates
#Probability of bitflip is sweeped between 10^(-4) and 0.32 in logscale and resulting P_success of the code is plotted
#Different lines are for different values of wait time between each rounds of error correction and number of total rounds.
#Total wait time is always 32units but each line represents a different way to dissect the correction in rounds
#A zoom of the graph is also produced to better show the behaviour of the code for low values of p_bitflip, where the code is more effective, 
#Here P_failure is plotted in logscale (instead of P_success) to highlight the differences between each way of dissecting the correction in rounds

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

points = 50
p_bf = np.logspace(-4,-0.5,points)
pteo_32u = []
pfail_teo = []

zoomb = 10**(-2.5) #zoom boundary
posiz = 0
for i in range(points):
    if p_bf[i] <= zoomb:
        posiz = i
    else:
        i=101

p_bf_fail = p_bf[0:posiz+1]

round = np.array([1,2,4,8,16,32])
dt = np.array([32,16,8,4,2,1])
psucc = np.zeros((6,points), dtype='float64')
pfail = np.zeros((6,posiz+1), dtype='float64')
col = -1

#delta_t = 2u
for riga in range(6):
    col = -1
    for p_bitflip in p_bf:
        col = col+1
        noise_model = get_noise(p_bitflip)
        aer_sim = AerSimulator(noise_model = get_noise(p_bitflip))  #initializes noise model with p_gate and p_meas from input

        #Initialization of the quantum registers: 5 data qubits, 4 error detection ancillas, 1 measure ancilla, 4 classical bits, 1 bypass bit
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

        for i in range (round[riga]):
            qc.reset(anc)
            for j in range (dt[riga]):
                qc.id(data).c_if(cbit, 0) #bypass the transpiler to compute identity

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
        okcounts = 0
        for i in x:
            if "0 0 " in i:
                okcounts += counts[i]

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
plt.suptitle("QEC: cyclic 5-qubit Steane code [5,1,3]", fontsize=20, fontname='Sans', fontweight='bold')
plt.xscale('log')
plt.legend(["with QEC rounds=1 dt=32u", "with QEC rounds=2 dt=16u", "with QEC r=4 dt=8u", "with QEC r=8 dt=4u", 
                "with QEC r=16 dt=2u","with QEC r=32 dt=1u", "1 qubit, no correction for dt=32u"], fontsize=14, loc='lower left')
#plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks([10**(-4), 10**(-3), 10**(-2), 10**(-1), 0.3])
plt.xlim([1e-04, 10**(-0.5)])
#plt.ylim([0.7,1])
plt.savefig("cyc5steane_Psucc_vs_Pbf.png", dpi=1000)
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
plt.suptitle("QEC: cyclic 5-qubit Steane code [5,1,3]", fontsize=20, fontname='Sans', fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.legend(["with QEC rounds=1 dt=32u", "with QEC rounds=2 dt=16u", "with QEC r=4 dt=8u", "with QEC r=8 dt=4u", 
                "with QEC r=16 dt=2u","with QEC r=32 dt=1u"], fontsize=14, loc='upper left')    #, "1 qubit, no correction for dt=32u"

plt.xticks([10**(-4), 10**(-3), zoomb])
plt.xlim([1e-04, zoomb])
plt.savefig("cyc5steane_Pfail_vs_Pbf.png", dpi=1000)
plt.show()