#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:17:06 2023

@author: reeseboucher
"""
import qiskit 

def oneBitFullAdder(qbit_a, qbit_b, carry_in):

  circuit = qiskit.QuantumCircuit(8,2)

  # flips qbit value to 1 if specified
  if int(qbit_a) == 1:
    circuit.x(0)
  if int(qbit_b) == 1:
    circuit.x(1)
  if int(carry_in) == 1:
    circuit.x(2)

  # one bit full adder quantum digital logic
  circuit.toffoli(0,1,3)
  circuit.cnot(0,4)
  circuit.cnot(1,4)
  circuit.cnot(2,5)
  circuit.cnot(4,5)
  circuit.toffoli(2,4,6)
  circuit.x(3)
  circuit.x(6)
  circuit.toffoli(3,6,7)
  circuit.x(7)

  # measure the circuit in its final state 
  circuit.measure(5,0) 
  circuit.measure(7,1) 

  return circuit

# all possible combinations in truth table 
permutationList = [["0","0","0"],["0","0","1"],["0","1","0"],["0","1","1"],["1","0","0"],["1","0","1"],["1","1","0"],["1","1","1"]]

# loop over possible permutations to find all possibilities
for permutation in permutationList:
    
  circuit    = oneBitFullAdder(permutation[0], permutation[1], permutation[2])
  runCircuit = qiskit.execute(circuit, qiskit.Aer.get_backend('qasm_simulator'))
  result     = runCircuit.result()
  counts     = result.get_counts()
  print("input(a,b,carry-in):",permutation)
  print("output(carry-out,sum):",next(iter(counts)))
  print("\n")
  
  