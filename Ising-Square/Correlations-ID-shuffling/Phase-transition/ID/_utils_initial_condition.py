import numpy as np
eps = 1E-7


def initial_condition(N,k,T,alphamin,alphamax,
                      # T_list0=T_list0,
                      # opt0folder=opt0folder,
                      ):
  """ This routine takes the initial condition of the previous temperature in 
  the optimization  of the non-shuffled system, to reduce variance."""

  ### Temperatures from the non-shuffle optimization
  T_list0 = np.arange(2,2.1+eps,.1)
  T_list0 = np.concatenate((T_list0,
                          np.arange(2.2,2.4+eps,.01))
                          )
  T_list0 = np.concatenate((T_list0,
                          np.arange(2.5,3.+eps,.1)
                          )
                          )
  T_list0 = np.flip(T_list0)
  opt0folder = f'../ID0/results/opt/'

  T_id = np.where(np.isclose(T,T_list0))[0][0]
  T = T_list0[T_id-1]
  # print(f'{T=}')
  opt0file = opt0folder + f'L{N}_k{k}_T{T:.2f}_alphamin{alphamin:.5f}_alphamax{alphamax:.5f}.txt'
  return np.loadtxt(opt0file,delimiter=',') # rmax,sigma,alpha,logKL