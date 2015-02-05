import numpy as np

Connection_mat=None
Spike_state=None

class neuron:
	def __init__(self,num_neurons,num_time_bins):
		self.coupling_mat = np.ones((num_neurons,num_neurons))
		self.phi = 0;
		self.X=np.zeros((0,0))

def init_Connection_mat(fluorescence,connectivity,clustering_coeff):
	number_of_neurons=len(fluorescence[0])
	number_of_time_bins = len(fluorescence[1])
	global Connection_mat
	Connection_mat=np.ones((number_of_neurons,number_of_neurons))
	#set C according to smart heuristics 

def init_Spike_state(fluorescence):
	#set B according to smart heuritics 
	number_of_neurons=len(fluorescence[0])
	number_of_time_bins = len(fluorescence[1])
	global Spike_state 
	Spike_state = np.zeros(number_of_neurons,number_of_neurons)



