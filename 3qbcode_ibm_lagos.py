# This code runs dynamical cyclic 3-qubit code bitflip error correction on IBM Quantum systems with 7 qubits; for example IBM_LAGOS
#Useful libraries:
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_provider import IBMProvider

#Suppress warnings
import warnings
warnings.filterwarnings('ignore')

#Select the provider
provider = IBMProvider()

print("(CYCLIC) 3 QUBIT REPETITION CODE QEC on IBM quantum hardware \n")

rs = 2   #number of qec rounds

rs = rs-1   #effective rounds are rs+1, this are just the rounds inside the 'for' loop

#Initialization of the quantum registers: 3 data qubits, 2 ancillas, 2 classical bits
qr = QuantumRegister(7)
sbit = ClassicalRegister(2)     #syndrome bits
mbit = ClassicalRegister(3)     #data measurement bits
tbit = ClassicalRegister(2)     #test bits

# In order to optimize the circuits and select the qubits with less error rates
# we map the physical qubits to the logical qubits directly using this index nomenclature:
initial_layout = [0, 1, 2, 3, 4, 5, 6]
data = [2, 3, 6]
anc = [1, 5]
test = [0, 4]
chain = [2, 1, 3, 5, 6]

qc = QuantumCircuit(qr, tbit, sbit, mbit)

# qc.h(qr[test[0]])   #example: I want to store the information of a |+> state in a single qubit
# qc.h(qr[test[1]])   #statistics with n=2 :/

# qc.h(qr[data[0]])   #example: I want to store the information of a |+> state in the logical qubit

# The following circuit encodes the first data qubit's state in all the data qubits and then measures their XX-stabilizers
# the circuit has minimum possible depth in chain-disposed qubits 
for i in range (len(chain)-1):
    qc.cx(qr[chain[i]], qr[chain[i+1]])

for i in range (len(chain)-1, 1, -2):
    qc.cx(qr[chain[i]], qr[chain[i-1]])

qc.measure(qr[anc[0]], sbit[0])     #ancilla is measured and the result stored in a classical bit
qc.measure(qr[anc[1]], sbit[1])

#print(qc)
#classical controlled dynimical bitflip-error correction
with qc.if_test((sbit, 1)):
    qc.x(qr[data[0]])
with qc.if_test((sbit, 3)):
    qc.x(qr[data[1]])
with qc.if_test((sbit, 2)):
    qc.x(qr[data[2]])


for i in range (rs):
    qc.reset(qr[anc[0]])
    qc.reset(qr[anc[1]])   #reset ancillas to |0>
           
    #entangle 2 couples of data-qubit to 2 ancillas to study internal correlation in the logical qubit
    qc.cx(qr[data[0]], qr[anc[0]])
    qc.cx(qr[data[1]], qr[anc[0]])
    qc.cx(qr[data[1]], qr[anc[1]])
    qc.cx(qr[data[2]], qr[anc[1]])
    # ancilla_0 shows if data_qubit_0 = data_qubit_1
    # ancilla_1 shows if data_qubit_1 = data_qubit_2

    #measure the ancillas to detect errors
    qc.measure(qr[anc[0]], sbit[0])     #ancilla is measured and the result stored in a classical bit
    qc.measure(qr[anc[1]], sbit[1])

    #if an error is detected in one of the qubits we apply a classically controlled not gate to correct it
    with qc.if_test((sbit, 1)):         #ATTENTION: '1' = '01' == 'sbit[1],sbit[0]' 
        qc.x(qr[data[0]])
    with qc.if_test((sbit, 3)):         #ATTENTION: 3 = '11' == 'sbit[1],sbit[0]'
        qc.x(qr[data[1]])
    with qc.if_test((sbit, 2)):         #ATTENTION: 2 = '10' == 'sbit[1],sbit[0]'
        qc.x(qr[data[2]])


qc.measure(qr[data[0]], mbit[0])    # measure the data qubits
qc.measure(qr[data[1]], mbit[1])
qc.measure(qr[data[2]], mbit[2])

qc.measure(qr[test[0]], tbit[0])    # measure the test qubits
qc.measure(qr[test[1]], tbit[1])

# Select a backend
backend = provider.get_backend("ibm_lagos")     #Between "" is the name of the chosen quantum computer

# Transpile the circuit
transpiled = transpile(qc, backend=backend, initial_layout=initial_layout)  #initial_layout contains all the on-circuit optimization information

# Submit a job
job = backend.run(transpiled, shots=7*1024, dynamic=True)

print(job.status())  #Hopefully Queued