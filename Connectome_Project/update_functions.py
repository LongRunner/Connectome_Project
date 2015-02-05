import io
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import os
from scipy.stats import norm
from scipy.stats import truncnorm
from collections import defaultdict

############################################
####  General-purpose functions
############################################

def read_data( file_location ):
  infile = open( file_location, 'r' )
  output = []
  for line in infile.readlines() :
    output += [ json.loads( '[' + line + ']' ) ]
  # print(np.array(output))
  return np.array( output )

def write_to_csv( file_location, data, names = None ):
  out = open( file_location, 'w' )
  if names :        out.write( json.dumps(names)[1:-1] + '\n' )
  for obj in data : out.write( json.dumps( obj )[1:-1] + '\n' )
  
  out.close()

def constant_factory(value):
  # Used for defaultdict's default value
  import itertools
  return itertools.repeat(value).next

############################################
####  Support Functions
############################################

def low_mem_selfdot( mat ):
  n = mat.shape[1]
  output = np.empty([ n, n ])
  for i in xrange(n) :
    for j in xrange(i, n) :
      output[i,j] = output[j,i] = mat[:,i].dot( mat[:,j] )
  
  return output[i,j]

def low_mem_cov( mat ):
  # assumes mat is pre-centered!
  return low_mem_selfdot( mat ) / ( mat.shape[0] + 0. )

def distance_mat( positions ):
  n = len(positions)
  mat = np.empty( [ n, n ] )
  for i in xrange(n-1) :
    for j in xrange(i, n) :
      mat[i,j] = mat[j,i] = sum(( positions[i,:] - positions[j,:] )**2)
  
  return mat

def Scores( C, Chat ):
  N = C.shape[0]
  Correct = (C * Chat).sum() + 0.
  Positives = (Chat).sum()
  Connections = C.sum()
  Precision = Correct / Positives
  Recall    = Correct / Connections
  F1 = 2 * Precision * Recall / (Precision + Recall)
  return [Precision, Recall, F1]

def gaussian_blur_matrix( positions, h = 0.15, A = 0.15 ) :
  n = len(positions)
  d = len(positions[0])
  mat = np.empty( [ n, n ] )
  for i in xrange(n-1) :
    for j in xrange(i, n) :
      if i == j :
        mat[i,i] = 1.
      else :
        mat[i,j] = mat[j,i] = A * np.exp( - sum(( positions[i,:] - positions[j,:] )**2 / h ** 2 ) )
  
  return mat

def C_from_network( net, N = None ) :
  if N == None : N = net[:,:1].max()
  C = np.diag( np.ones([N]) )
  for i in xrange(net.shape[0]) :
    if net[i,2] == 1 :
      C[ net[i,0]-1, net[i,1]-1 ] = 1
  
  return C

# C_from_network( net ).sum()

############################################
####  Model building
############################################

def Update_C( Beta, rho, Tau, Stochastic = False ) :
  N, M, L = Beta.shape
  LogOdds = .5 * ( L * np.log( Tau[1] / Tau[0] ) - (Tau[1] - Tau[0]) * (Beta ** 2 ).sum( 2 ) )
  LogOdds += np.log( rho / (1 - rho ) )
  if Stochastic == True :
    Check = -1. * np.log( (1/np.random.uniform( size = [N,N] ) - 1) )
    Sample = ( Check < LogOdds )
  else :
    Sample = (   0   < LogOdds )
  
  for i in xrange(N) : Sample[i,i] = True
  return Sample

# Update_C( np.random.normal( size = [3, 3, 10] ), .5, [ 1., 1. ], True).sum() - 3
# Should be binomial( 6, 1/2 )

def Update_BetaRow( n ) :
  Xt = np.append( Ft, Ft * F[:,n], 0)[:,:-1]
  TauMat = Var * np.diag( np.tile( Tau[0] + (Tau[1] - Tau[0]) * C[0,:], [2,1] ).transpose().flatten() )
  XtXTaui = np.linalg.inv( Xt.dot(Xt.transpose()) + TauMat )
  BetaHat = XtXTaui.dot( Xt.dot( F[1:,n] ) )
  if Stochastic == True :
    eps = np.random.multivariate_normal( np.zeros([2*N]), XtXTaui )
    BetaHat += eps
  
  Fpred = BetaHat.dot( Xt )
  return [ BetaHat, Fpred ]

def Update_Beta_Mapped( F, Tau, C, Var, Stochastic = False ):
  T, N = F.shape 
  Ft = F.transpose()
  def Update_BetaRow( n ) :
    Xt = np.append( Ft, Ft * F[:,n], 0)[:,:-1]
    TauMat = Var * np.diag( np.tile( Tau[0] + (Tau[1] - Tau[0]) * C[0,:], [2,1] ).transpose().flatten() )
    XtXTaui = np.linalg.inv( Xt.dot(Xt.transpose()) + TauMat )
    BetaHat = XtXTaui.dot( Xt.dot( F[1:,n] ) )
    if Stochastic == True :
      eps = np.random.multivariate_normal( np.zeros([2*N]), XtXTaui )
      BetaHat += eps
    
    Fpred = BetaHat.dot( Xt )
    return [ BetaHat, Fpred ]
  
  map_result = map( Update_BetaRow, range(N) )
  BetaHat = np.array([ result[0] for result in map_result ])
  Fpred = np.array([ result[1] for result in map_result ]).transpose()
  return [ BetaHat, Fpred ]

Beta, Fpred = Update_Beta_Mapped( Fbl, [100., .01], np.random.uniform(size = [N,N]), 1., Stochastic = False )

def Update_Beta( F, Tau, C, Var, Stochastic = False ):
  T, N = F.shape
  Ft = F.transpose()
  BetaHat = np.empty([N, 2*N])
  Fpred = np.empty([T-1, N])
  
  # This for loop is embarrassingly parallelizable
  for n in range(N) :
    Xt = np.append( Ft, Ft * F[:,n], 0)[:,:-1]
    TauMat = Var * np.diag( np.tile( Tau[0] + (Tau[1] - Tau[0]) * C[0,:], [2,1] ).transpose().flatten() )
    XtXTaui = np.linalg.inv( Xt.dot(Xt.transpose()) + TauMat )
    BetaHat[n,:] = XtXTaui.dot( Xt.dot( F[1:,n] ) )
    if Stochastic == True :
      eps = np.random.multivariate_normal( np.zeros([2*N]), XtXTaui )
      BetaHat[n,:] += eps
    
    Fpred[:,n] = BetaHat[n,:].dot( Xt )
  
  return [ BetaHat, Fpred ]

# Update_Beta( Fbl, [100., .01], np.random.uniform(size = [N,N]), 1., Stochastic = False )

def Update_A( Fbl, F, DistMat, 
              a0 = .5, lambda0 = .15, samples = 10, Var = 1.,
              Mode = ['MLE', 'Expectation', 'Stochastic'][0] ) :
  T, N = F.shape
  alphas  =      a0 * np.exp(np.random.normal( size = samples ))
  lambdas = lambda0 * np.exp(np.random.normal( size = samples ))
  
  # This list comprehension is also embarrassingly parallelizable 
  SSmat = np.array([ [ ((Fbl - F.dot( a * np.exp(- DistMat / l**2) + np.diag(np.tile([1-a],[N])) ) )**2).sum()
                        for l in lambdas ] for a in alphas ])
  
  Emat = -.5 * (SSmat - SSmat.max()) / Var
  pmat = np.exp( Emat - Emat.max() )
  pmat = pmat / pmat.sum()
  
  if Mode == 'MLE' :
    alpha_star = alphas[ list(SSmat.min(1)).index(SSmat.min()) ]
    lambda_star = lambdas[ list(SSmat.min(0)).index(SSmat.min()) ]
  elif Mode == 'Stochastic':
    alpha_star = alphas[ ( np.random.rand() < pmat.sum(1).cumsum() / SSmat.sum() ).sum() - 1 ]
    lambda_star = lambdas[ ( np.random.rand() < pmat[ alphas.index(alpha_star) ].cumsum() / pmat.sum() ).sum() - 1 ]
  elif Mode == 'Expectation' :
    alpha_star  = pmat.transpose().dot(alphas).sum()
    lambda_star = pmat.dot(lambdas).sum()
  else :
    alpha_star = .5;
    lambda_star = .5;
  
  A_star = alpha_star * np.exp( - DistMat / lambda_star**2) + np.diag(np.tile([1-a],[N]))
  return [ A_star, alpha_star, lambda_star ]

# Dmat = distance_mat( pos )
# Update_A( Fbl, Fbl * .8 + .2 * np.random.normal(size = [ T, N ]), Dmat, Mode = 'MLE' )

def Update_Var( F, Fhat, Prior_a = 1e-4, Prior_b = 1e4 ) :
  T, N = F.shape
  SS = ((F - Fhat)**2).sum()
  Var = np.random.gamma( Prior_a + N/2., 1 / ( 1./Prior_b + SS / 2. ) )
  return Var

# Update_Var( Fbl, .8 * Fbl, 10., 10. )
