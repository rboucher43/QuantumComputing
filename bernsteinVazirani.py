#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:52:19 2023

@author: reeseboucher

Implementation of the Bernstein-Vazirani Algorithm
"""

import qiskit

def bernstein_vazirani(secretString, qbits):
    
    circuit = qiskit.QuantumCircuit(qbits+1, qbits) # create circuit 
    
    circuit.h(qbits) # hadamard gate
    circuit.z(qbits) # z gate 
        
    secretString = secretString[::-1] 
    
    # loop over all qbits and apply desired operations
    for qbit in range(qbits):
        circuit.h(qbit)
        if secretString[qbit] == '0':
            circuit.id(qbit) # identity gate 
        else:
            circuit.cnot(qbit, qbits)
        circuit.h(qbit)
        circuit.measure(qbit, qbit)
    
    # simulate circuit run and print results
    simulator    = qiskit.Aer.get_backend('aer_simulator')
    outputString = simulator.run(circuit).result().get_counts()
    print("Output Secret String:",outputString)    
    print(circuit)
    
    
inputString  = "010"  # Secret string that will be input to bernstein_vazirani method
qbits        = len(inputString)      # number of qbits in circuit
bernstein_vazirani(inputString, qbits)
