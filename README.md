# qiskit_thesis
Python codes to perform Quantum Error Correction on real hardware and simulators using Qiskit.
'_graph' files produce the plots of each algorithm simulated success probabilities as a function of the noise model's parameter (error probability), swiped in logscale.
'bf' stands for bitflip noise model, 'gate' for depolarizing noise model. 'code_comparison' has fixed identity bitflip probability and swipes depolarizing error probability.
'_all' files are the most general simulation of each code which lets you choose the number of QEC rounds, the wait time between rounds and the noise model parameters.
'dynamical_nqubitcode.py' is the simulation for any size of the Repetition code. You can again choose the number of rounds, the wait time between rounds and error probabilities. This program also contains an efficient
way to calculate all possible syndroms (Lookup table definition).
'_ibm_' files launch the codes on IBM Quantum processors. The one of IBM Cairo also uses the efficient Lookup table definition.
'job_retriever' collects and analyzes the results of real hardware. 
