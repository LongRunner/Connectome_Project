############################################
####  Function loading
############################################

# project_folder = "/Users/guywcole/Desktop/Dropbox/CollegeStuff/Spring14/NC/NeuralCoding/Connectome_Project/"
project_folder = "/Volumes/Excess/Dropbox/CollegeStuff/Spring14/NC/NeuralCoding/Connectome_Project/"
import os
os.chdir(project_folder)

#exec(open('ism_guy_functions.py').read())
execfile('guys_functions.py')

############################################
####  Data loading
############################################

Fbl = read_data(     'fluorescence_iNet1_Size100_CC03inh.txt' )
net = read_data(          'network_iNet1_Size100_CC03inh.txt' )
pos = read_data( 'networkPositions_iNet1_Size100_CC03inh.txt' )

############################################
####  Initialization
############################################
T, N = Fbl.shape
Tau = [100, .01]
Gamma_a = 200
Gamma_b = 1
C = np.ones([N,N])
alpha_star = .15
lambda_star = .15
Fstar = Fbl.dot( np.linalg.inv( gaussian_blur_matrix( pos, alpha_star, lambda_star ) ) )
Var = 0.002
DistMat = distance_mat( pos )
rho = 0.10
Ctrue = C_from_network( net )

############################################
####  Updating
############################################
iters = 10
print [-1] + Scores( Ctrue, C )

for q in xrange(iters) : 
  Beta, Fpred = Update_Beta( Fstar, Tau, C, Var, Stochastic = True )
  C = Update_C( Beta.reshape(N,N,2), rho, Tau, Stochastic = True )
  Var = Update_Var( Fstar[1:,:], Fpred, Gamma_a, Gamma_b )
  A, alpha_star, lambda_star = Update_A( Fbl[1:,:], Fpred, DistMat, alpha_star, lambda_star, 5, Var, 'MLE' )
  Fstar = Fbl.dot( np.linalg.inv( A ) )
  print [q] + Scores( Ctrue, C )

for q in xrange(iters) : 
  Beta, Fpred = Update_Beta( Fbl, Tau, C, Var, Stochastic = True )
  C = Update_C( Beta.reshape(N,N,2), rho, Tau, Stochastic = True )
  Var = Update_Var( Fbl[1:,:], Fpred, Gamma_a, Gamma_b )
  print [q] + Scores( Ctrue, C )

