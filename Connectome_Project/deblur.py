from collections import defaultdict
import math
import numpy as np
import brainparse
import io
import json
import os

lamda_sc = .15 #referenced from the Stetter paper 
A_sc = .15 #referenced from the Stetter paper

def deblur():
	"""Iterate through and remove noise per specifications in the Materials and Methods section of Model-Free Reconstruction of Excitatory Neuronal
    Connectivity from Calcium Imaging Signals by Olav Stetter.

	neuron_positions --- dictionary with dict[neuron_id] = (x,y) where x and y are scaled to 500 (from 1000)
	neuron_time_series --- dictionary with dict[neuron_id]=[activity_at_1, ... n]

	"""
	#we might want these as global parameters just for convenience. i have it this way currently for debugging purposes.
	neuron_time_series = brainparse.read_data('/Users/austinstone/code/NeuralCodingFinalProject/small/fluorescence_iNet1_Size100_CC03inh.txt')
	neuron_positions = brainparse.read_data('/Users/austinstone/code/NeuralCodingFinalProject/small/networkPositions_iNet1_Size100_CC03inh.txt')
	number_of_neurons=len(neuron_positions)
	number_of_time_bins = len(neuron_time_series)
	deblurred_fluorescence = np.zeros((number_of_time_bins,number_of_neurons))
	write_to_csv('test.csv', neuron_time_series)	
	dist_mat = get_distance_mat(neuron_positions,A_sc,lamda_sc)
	for time_bin in range(0,number_of_time_bins):
		#print time_bin
     		deblurred_fluorescence[time_bin] = np.linalg.solve(dist_mat,neuron_time_series[time_bin])
		#print(deblurred_fluorescence_vec)
	write_to_csv('/Users/austinstone/code/NeuralCodingFinalProject/small/deblurred_fluorescence_NW3.csv', deblurred_fluorescence)	
	return deblurred_fluorescence


def get_distance_mat(neuron_positions,A,h):
	number_of_neurons = len(neuron_positions)
	dist_mat = np.zeros((number_of_neurons,number_of_neurons))
	for i in range(0,number_of_neurons):
		for j in range(0,number_of_neurons):
			if(i==j):
				dist_mat[i][j]=1
			else:
				dist_mat[i][j]=A*math.exp(0-(brainparse.calc_dist(neuron_positions[i],neuron_positions[j])/h)**2)
	return dist_mat


def write_to_csv( file_location, data, names = None ):
	np.savetxt(file_location, data, delimiter=",")




if __name__ == '__main__':
    import sys
    deblur()

