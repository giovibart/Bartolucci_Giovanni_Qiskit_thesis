# Description: This program retrieves the results of a job from IBM Quantum Hardware and prints the probability of success
# and the Kullback-Leibler divergence for the n-Qubit code. You can input the size of the code (n) and the initial state.
# The program also calculates the success probabilities of for 4 test qubits left idling during the code.
#The results must be in order: Test A, Test B, Test C, Test D, Data qubits, ...
#The program is designed to work with IBM Cairo but is easily adaptable to other devices changing the number of test qubits.

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_provider import IBMProvider
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import math

provider = IBMProvider(instance='instance_path')
job_name = 'job_id'
job_finished = provider.retrieve_job(job_name)
counts = job_finished.result().get_counts()

#print(counts)
print("\n JOB RETRIEVER FOR CYCLIC n-QUBIT CODE ON IBM QUANTUM HARDWARE\n")
n = input("Insert code size (number of data qubits) \n")
bs = input("Insert the number corresponding to the initial state: '0'<->|0>, '1'<->|1>, '2'<->|+>\n")
bs = int(bs)
n = int(n)

x = counts.keys()

data0 = 0
data1 = 0
ta0=0
ta1=0
tb0=0
tb1=0
tc0=0
tc1=0
td0=0
td1=0
tot=0


for i in x:
    tot += counts[i]
    if (" "+'0'*n+" ") in i:
        data0 += counts[i]
    if (" "+'1'*n+" ") in i:
        data1 += counts[i]
    if i[0]=='0':
        ta0 += counts[i]
    if i[0]=='1':
        ta1 += counts[i]
    if i[1]=='0':
        tb0 += counts[i]
    if i[1]=='1':
        tb1 += counts[i]
    if i[2]=='0':
        tc0 += counts[i]
    if i[2]=='1':
        tc1 += counts[i]
    if i[3]=='0':
        td0 += counts[i]
    if i[3]=='1':
        td1 += counts[i]

print("shots = ", tot)

pd0 = data0/tot
pd1 = data1/tot
pta0 = ta0/tot
pta1 = ta1/tot
ptb0 = tb0/tot
ptb1 = tb1/tot
ptc0 = tc0/tot
ptc1 = tc1/tot
ptd0 = td0/tot
ptd1 = td1/tot

if bs==0:
    print('\nP_success=P_|0...0> = {}%'.format(round(100*pd0,2)))
    D_KL = -math.log2(pd0)
    print('D_KL = {} '.format(round(D_KL,4)))
    print('\nP_testA_succ=P_|0> = {}%'.format(round(100*pta0,2)))
    D_KL_ta = -math.log2(pta0)
    print('D_KL_testA = {} '.format(round(D_KL_ta,4)))
    print('P_testB_succ=P_|0> = {}%'.format(round(100*ptb0,2)))
    D_KL_tb = -math.log2(ptb0)
    print('D_KL_testB = {} '.format(round(D_KL_tb,4)))
    print('P_testC_succ=P_|0> = {}%'.format(round(100*ptc0,2)))
    D_KL_tc = -math.log2(ptc0)
    print('D_KL_testC = {} '.format(round(D_KL_tc,4)))
    print('P_testD_succ=P_|0> = {}%'.format(round(100*ptd0,2)))
    D_KL_td = -math.log2(ptd0)
    print('D_KL_testD = {} '.format(round(D_KL_td,4)))

if bs==1:
    print('\nP_success=P_|1...1> = {}%'.format(round(100*pd1,2)))
    D_KL = '+inf'
    if(pd1!=0):
        D_KL = -math.log2(pd1)
        D_KL = round(D_KL,4)
    print('D_KL = ', D_KL)
    print('\nP_testA_succ=P_|1> = {}%'.format(round(100*pta1,2)))
    D_KL_ta = -math.log2(pta1)
    print('D_KL_testA = {} '.format(round(D_KL_ta,4)))
    print('P_testB_succ=P_|1> = {}%'.format(round(100*ptb1,2)))
    D_KL_tb = -math.log2(ptb1)
    print('D_KL_testB = {} '.format(round(D_KL_tb,4)))
    print('P_testC_succ=P_|1> = {}%'.format(round(100*ptc1,2)))
    D_KL_tc = -math.log2(ptc1)
    print('D_KL_testC = {} '.format(round(D_KL_tc,4)))
    print('P_testD_succ=P_|1> = {}%'.format(round(100*ptd1,2)))
    D_KL_td = -math.log2(ptd1)
    print('D_KL_testD = {} '.format(round(D_KL_td,4)))

if bs==2:
    print('\nP_|0...0> = {}%'.format(round(100*pd0,2)))
    print('P_|1...1> = {}%'.format(round(100*pd1,2)))
    D_KL = '+inf'
    p_succ = 0
    if(pd1!=0 and pd0!=0):
        D_KL = 0.5*math.log2(0.5/pd0)+0.5*math.log2(0.5/pd1)
        p_succ = 2**(-D_KL)
        D_KL = round(D_KL,4)
    print('D_KL = ', D_KL)
    print('\nP_success = 2^(-D_KL) = {}%'.format(round(100*p_succ,2)))
    D_KL_ta = 0.5*math.log2(0.5/pta0)+0.5*math.log2(0.5/pta1)
    print('D_KL_testA = {} '.format(round(D_KL_ta,4)))
    print('P_testA_succ = 2^(-D_KL_testA) = {}%'.format(round(100*2**(-D_KL_ta),2)))
    D_KL_tb = 0.5*math.log2(0.5/ptb0)+0.5*math.log2(0.5/ptb1)
    print('D_KL_testB = {} '.format(round(D_KL_tb,4)))
    print('P_testB_succ = 2^(-D_KL_testB) = {}%'.format(round(100*2**(-D_KL_tb),2)))
    D_KL_tc = 0.5*math.log2(0.5/ptc0)+0.5*math.log2(0.5/ptc1)
    print('D_KL_testC = {} '.format(round(D_KL_tc,4)))
    print('P_testC_succ = 2^(-D_KL_testC) = {}%'.format(round(100*2**(-D_KL_tc),2)))
    D_KL_td = 0.5*math.log2(0.5/ptd0)+0.5*math.log2(0.5/ptd1)
    print('D_KL_testD = {} '.format(round(D_KL_td,4)))
    print('P_testD_succ = 2^(-D_KL_testD) = {}%'.format(round(100*2**(-D_KL_td),2)))

