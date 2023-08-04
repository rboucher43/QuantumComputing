#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:23:47 2023

@author: reeseboucher
"""
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer, transpile, assemble, QuantumCircuit, QuantumRegister
from qiskit.visualization import plot_histogram


def generate_random_graph(num_nodes):
    complete_graph = nx.complete_graph(num_nodes)
    
    # Remove random edges to achieve 50% density
    edges_to_remove = random.sample(complete_graph.edges(), len(complete_graph.edges()) // 2)
    random_graph    = complete_graph.copy()
    random_graph.remove_edges_from(edges_to_remove)
    
    # Assign random weights to the remaining edges
    for edge in random_graph.edges():
        random_graph.edges[edge]['weight'] = random.random()
    
    return random_graph


def max_cut_hamiltonian(graph):
    num_nodes   = graph.number_of_nodes()
    pauli_z     = np.array([[1, 0], [0, -1]])
    hamiltonian = np.zeros((2 ** num_nodes, 2 ** num_nodes))
    
    for edge in graph.edges():
        i, j         = edge
        weight       = graph[i][j]['weight']
        hamiltonian -= (weight / 2) * np.kron(np.kron(np.eye(2 ** i), pauli_z), np.eye(2 ** (num_nodes - i - 1)))
        hamiltonian -= (weight / 2) * np.kron(np.kron(np.eye(2 ** j), pauli_z), np.eye(2 ** (num_nodes - j - 1)))
    
    return hamiltonian


def create_qaoa_circuit(graph, num_layers):
    num_nodes   = graph.number_of_nodes()
    hamiltonian = max_cut_hamiltonian(graph)
    q           = QuantumRegister(num_nodes, name='q')
    qc          = QuantumCircuit(q)

    qc.h(q)

    # Apply the QAOA circuit
    for layer in range(num_layers):
        for edge in graph.edges():
            i, j = edge
            qc.cp(-2 * hamiltonian[i, j], q[i], q[j])
        qc.barrier()

    qc.measure_all()

    return qc


def max_cut_value(graph, cut):
    cut_value = 0
    for edge in graph.edges():
        if cut[edge[0]] != cut[edge[1]]:
            cut_value += graph[edge[0]][edge[1]]['weight']
    return cut_value


def exhaustive_max_cut(graph):
    max_cut = 0
    for i in range(2 ** graph.number_of_nodes()):
        binary_str = format(i, f"0{graph.number_of_nodes()}b")
        cut        = [int(bit) for bit in binary_str]
        cut_value  = max_cut_value(graph, cut)

        if cut_value > max_cut:
            max_cut = cut_value

    return max_cut



def get_max_cut_qaoa(counts, graph):
    max_cut = 0
    for cut_str, frequency in counts.items():
        cut       = [int(bit) for bit in cut_str[::-1]] 
        cut_value = max_cut_value(graph, cut)
        if cut_value > max_cut:
            max_cut     = cut_value

    return max_cut


def plot_graph(graph):
    pos         = nx.spring_layout(graph, seed=42)  
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    
    nx.draw(graph, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Max-Cut Graph")
    plt.show()


ex_max_cut_list = []
qa_max_cut_list = []
plot_bool    = False
nodes        = 8
num_layers   = 2


for i in range(1):
    # Generate a random graph with n nodes
    graph     = generate_random_graph(nodes)
    
    if plot_bool == True:
        plot_graph(graph)
      
    # Run the QAOA circuit on a simulator
    simulator    = Aer.get_backend('qasm_simulator')
    qaoa_circuit = create_qaoa_circuit(graph, num_layers)
    if plot_bool == True:
        print(qaoa_circuit)
    
    # Execute the circuit
    t_qaoa      = transpile(qaoa_circuit, simulator)
    qobj_qaoa   = assemble(t_qaoa)
    result_qaoa = simulator.run(qobj_qaoa).result()
    counts_qaoa = result_qaoa.get_counts(qaoa_circuit)
    
    if plot_bool == True:
        plot_histogram(counts_qaoa)
               
    print("Plot ",i)
    # QAOA search
    max_cut_qaoa = get_max_cut_qaoa(counts_qaoa, graph)
    print("QAOA Max-Cut Value:       ", max_cut_qaoa)
    
    # Exhaustive search
    max_cut_exhaustive = exhaustive_max_cut(graph)
    print("Exhaustive Max-Cut Value: ", max_cut_exhaustive)
    
    if max_cut_qaoa != max_cut_exhaustive:
        print("Max-cut values not equal")
        
    ex_max_cut_list.append(max_cut_exhaustive)
    qa_max_cut_list.append(max_cut_qaoa)
    
    


if plot_bool == True:
    # plot max-cut values
    plt.plot(ex_max_cut_list,qa_max_cut_list,".")
    plt.xlabel("Exhaustive Max-Cut Value")
    plt.ylabel("QAOA Max-Cut Value")
    plt.title("Max-Cut Values")







