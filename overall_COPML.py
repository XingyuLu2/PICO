from mpi4py import MPI
import numpy as np
import random
from array import array
import math
import time
import sys
import gc
import os
import matplotlib as mpl
import matplotlib.pylab as plt
import pickle as pickle

from utils.mpc_function import *
from utils.polyapprox_function import *
from datasets.Load_CIFAR10 import *


#################################################
############ Distributed System Setting #########
#################################################

# system parameters -- for communication client distribution
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) == 1:
    if rank ==0:
        print("ERROR: please input the number of workers")
    exit()
else:
    N = int(sys.argv[1])


#################################################
################# Learning parameters ###########
#################################################

# the total number of trainng iterations
max_iter = 50
# set the seed of the random number generator for consistency
np.random.seed(42)
# the value of prime
p, q_bit_X, q_bit_y = 2**26-5, 1, 0  
# the parameters of secure truncation
alpha_exp = 15
coeffs0_exp = 1
coeffs1_exp = 6
trunc_scale = alpha_exp + coeffs1_exp - q_bit_y
trunc_k, trunc_m = 24, trunc_scale
# the bandwidth in bits
BW = 40_000_000 # 40Mbps

######################################################
################# Distributor -- rank-0 ##############
######################################################

if rank == 0:
    print("This is the server clients ... ")
    
    print("00.Load in and Process the CIFAR-10 dataset.")

    # the particular case of (K, T) pair
    K = int(np.floor((N-1)/float(3))) + 1 - int(np.floor((N-3)/float(6)))
    T = int(np.floor((N-3)/float(6)))
    print("########### For the Case : K = ", K, " and T = ", T, " #########")
    m = int(9019*2)
    d = int(3073)
    print('01.Data conversion: real to finite field')
    m_ = 0
    for j in range(1, N+1): 
        # split the original dataset equally      
        m_j = int(m / N)
        start_row = int(m_j*(j-1))
        if j == N:
            m_j += (m%j)
        m_j = m_j - (m_j%K)
        m_ += m_j
        d = 3073
        d = (N-T)*K*(int(d/((N-T)*K))+1)
        # communicate the local dataset to the correspnding client
        comm.send(m_j, dest=j)
        comm.send(d, dest=j) 
    for j in range(1, N+1):
        comm.send(m_, dest=j)

    print('02. Random matrix and corresponding SS generation')
    comm.Barrier()

######################################################
################### Clients -- rank-j ################
######################################################

elif rank <= N:

    print("### This is the client-", rank, " ####")

    # the particular case of (K, T) pair
    K = int(np.floor((N-1)/float(3))) + 1 - int(np.floor((N-3)/float(6)))
    T = int(np.floor((N-3)/float(6)))

    ###################### Receive the dataset and random matrices #############

    m_i = comm.recv(source=0) # number of rows =  number of training samples
    d = comm.recv(source=0) # number of columns  = number of features
    m = comm.recv(source=0)
    
    X_i = np.random.randint(p, size=(m_i, d)) 
    y_scale = np.random.randint(p, size=(m_i, 1)) # coded matrix

    R_LCC_SS_T = np.random.randint(p, size=(T, int(m/K), d))
    r_LCC_SS_T = np.random.randint(p, size=(T,d,1))
    r1_SS_T = np.random.randint(p, size=(d,1))
    r2_SS_T = np.random.randint(p, size=(d,1))
    
    comm.Barrier()      

    ############################################
    #       Preprocessing Starts Here.         #
    ############################################

    ### Group setting for LCC encoding & decoding
    # each group has (T+1) clients
    if np.mod(N,T+1) == 0:
        group_id = int(int(rank - 1)/int(T+1))
        group_idx_set = range(group_id*(T+1), (group_id+1)*(T+1))
    else:
        group_id = int(int(rank - 1)/int(T+1))
        last_group_id = int(int(N)/int(T+1))
        if (group_id == last_group_id) | (group_id == last_group_id - 1):
            group_idx_set = range((last_group_id-1)*(T+1), N)
        else:
            group_idx_set = range(group_id*(T+1), (group_id+1)*(T+1))
    group_stt_idx = group_idx_set[0]
    group_idx_set_others = [idx for idx in group_idx_set if rank-1 != idx]
    my_worker_idx = rank - 1
    # end of group setting

    total_time = time.time()
    ### Preprocessing 1.1. Secret Share for X and y
    # compute the SS of the local dataset (images + labels)
    X_SS_T_i = SSS_encoding(X_i,N,T,p)
    y_SS_T_i = SSS_encoding(y_scale,N,T,p)
    # communicate the SS of X_i for [X]_i, and the SS of y_scale_i for [y]_i
    X_SS_T = []
    y_SS_T = []
    for j in range(1, N+1):
        if rank == j:
            X_SS_T.append(X_SS_T_i[j-1,:,:])
            y_SS_T.append(y_SS_T_i[j-1,:,:])
            for j_others in range(1, N+1):
                if j_others == rank: continue
                # send the size of the local dataset
                comm.send(m_i, dest=j_others)
                # send the SS of the local dataset -- images
                comm_volume += (m_i*d * 64)
                comm.Send(np.reshape(X_SS_T_i[j_others-1,:,:], m_i*d), dest=j_others)
                # send the SS of the local dataset -- labels
                comm_volume += (m_i*1 * 64)
                comm.Send(np.reshape(y_SS_T_i[j_others-1,:,:], m_i*1), dest=j_others)
            X_SS_T_i = None
            y_SS_T_i = None
        else:
            m_j = comm.recv(source=j)
            # receive the SS of data points
            data_SS = np.empty(m_j*d, dtype='int64')
            comm.Recv(data_SS, source=j)
            # receive the SS of labels
            label_SS = np.empty(m_j*1, dtype='int64')
            comm.Recv(label_SS, source=j)
            #reshape to the normal shape
            data_SS = np.reshape(data_SS, (m_j, d)).astype('int64')
            label_SS = np.reshape(label_SS, (m_j, 1)).astype('int64')
            X_SS_T.append(data_SS)
            y_SS_T.append(label_SS)
    X_SS_T = np.concatenate(np.array(X_SS_T))
    y_SS_T = np.concatenate(np.array(y_SS_T))

    ### Preprocessing 1.2. Initialize the whole model
    # compute the SS of the local dataset (images + labels)
    w_init_i = (1/float(m))*np.random.rand(d,1)
    w_init_i = my_q(w_init_i, 0, p)
    w_SS_T_i = SSS_encoding(w_init_i,N,T,p)

    w_SS_T = np.empty((N,d,1), dtype='int64')
    for j in range(1, N+1):
        if rank == j:
            w_SS_T[j-1,:,:] = w_SS_T_i[j-1,:,:]
            for j_others in range(1, N+1):
                if j_others == rank: continue
                # send the SS of the local initial model
                comm_volume += (d*1 * 64)
                comm.Send(np.reshape(w_SS_T_i[j_others-1,:,:], d*1), dest=j_others)
            w_SS_T_i = None
        else:
            data = np.empty(d*1, dtype='int64')
            comm.Recv(data, source=j)
            data = np.reshape(data, (d, 1)).astype('int64')
            w_SS_T[j-1,:,:] = data
    # compute the SS of the whole initialized model
    w_SS_T = np.mod(w_SS_T.sum(axis=0), p)
    
    ### Preprocessing 2.  LCC encoding of X
    '''
    input  : X_SS_T (=secret share of X= [X]_i)  
    output : X_LCC (=\widetiled{X}_i)        
    '''
    # 1.1. generate the secret share of encoded X 
    X_LCC_T = LCC_encoding_w_Random_partial(X_SS_T,R_LCC_SS_T,N,K,T,p,group_idx_set)
    # 1.2. sending the secret share of encoded X
    dec_input = np.empty((len(group_idx_set), (int(m/K))*d), dtype='int64')
    for j in group_idx_set:
        if my_worker_idx == j:
            dec_input[my_worker_idx - group_stt_idx,:] = np.reshape(X_LCC_T[my_worker_idx - group_stt_idx,:,:], (int(m/K))*d )
            for idx in group_idx_set_others:
                data = np.reshape(X_LCC_T[idx - group_stt_idx,:,:], (int(m/K))*d )
                comm_volume += ((m/K)*d * 64)
                comm.Send(data, dest=idx+1) # sent data to worker j
            X_LCC_T = None
        else:
            data = np.empty((int(m/K))*d, dtype='int64')
            comm.Recv(data, source=j+1)
            dec_input[j-group_stt_idx,:] = data # coefficients for the polynomial
    # 1.3.  reconstruct the secret : get X_LCC
    X_LCC_dec = SSS_decoding(dec_input, group_idx_set, p)
    X_LCC = np.reshape(X_LCC_dec, (int(m/K), d)).astype('int64')
    
    ### Preprocessing 3. Calculate common terms
    ## compute the X.T*X_LCC
    XTX_LCC = X_LCC.T.dot(X_LCC)
    ## compute the [X.T*y]_i for client-i -- Degree Reduction
    # generate the U matrix
    U_matrix = np.empty((int(2*T+1), N), dtype="int64")
    for t in range(int(2*T+1)):
        if t == 0:
            U_matrix[t,:] = np.ones(N, dtype="int64")    
        else:
            # modular operation for Overflow problem
            U_matrix[t,:] = np.mod(np.array([i**t for i in range(1, N+1)]), p)
    U_inv = np.linalg.pinv(U_matrix)
    
    XTy_a_i = np.mod(X_SS_T.T.dot(y_SS_T) * U_inv[rank-1][0], p)
    XTy_a_i_SS = SSS_encoding(XTy_a_i,N,T,p)
    # communicates the SS of the term : [XTy]_i * a_i1
    XTy_SS_T = np.empty((N,d,1), dtype='int64')
    for j in range(1, N+1):
        if rank == j:
            XTy_SS_T[j-1,:,:] = XTy_a_i_SS[j-1,:,:]
            for j_others in range(1, N+1):
                if j_others == rank: continue
                # send the SS of the local initial model
                comm_volume += (d*1 * 64)
                comm.Send(np.reshape(XTy_a_i_SS[j_others-1,:,:], d*1), dest=j_others)
            XTy_a_i_SS = None
        else:
            data = np.empty(d*1, dtype='int64')
            comm.Recv(data, source=j)
            data = np.reshape(data, (d, 1)).astype('int64')
            XTy_SS_T[j-1,:,:] = data
    XTy_SS_T = np.mod(XTy_SS_T.sum(axis=0), p)

    ############################################
    #           Main Loop Starts Here.         #
    ############################################

    # set parameters
    iter = 0
    while (iter < max_iter):
        iter = iter + 1

        ### 1. LCC encoding of w(t)
        '''
        input  : w_SS_T (=secret share of w(t)= [w(t)]_i)  
        output : w_LCC (=\widetiled{w}^{(t)}_i)
        '''
        # 1.1 generate the secret share of encoded w 
        # repeated vector with size ( d*K by 1 )
        w_rep_SS_T = np.transpose(np.tile(np.transpose(w_SS_T), K)) 
        w_LCC_SS_T = LCC_encoding_w_Random_partial(w_rep_SS_T,r_LCC_SS_T,N,K,T,p,group_idx_set)
        # 1.2. sending the secret share of encoded w
        dec_input = np.empty((len(group_idx_set), d), dtype='int64')
        for j in group_idx_set:
            if my_worker_idx == j:
                dec_input[my_worker_idx - group_stt_idx,:] = np.reshape(w_LCC_SS_T[my_worker_idx - group_stt_idx,:,:], d )
                for idx in group_idx_set_others:
                    #print 'from',rank,' to ',idx+1
                    data = np.reshape(w_LCC_SS_T[idx - group_stt_idx,:,:], d )
                    comm_volume += (d*1 * 64)
                    comm.Send(data, dest=idx+1) # sent data to worker j                        
            else:
                data = np.empty(d, dtype='int64')
                comm.Recv(data, source=j+1)
                dec_input[j-group_stt_idx,:] = data # coefficients for the polynomial
        # 1.3. reconstruct the secret : get w_LCC
        w_LCC_dec = SSS_decoding(dec_input, group_idx_set, p) 
        w_LCC = np.reshape(w_LCC_dec, (d, 1)).astype('int64')

        ### 2. compute f over LCC_encoded inputs
        f_eval = np.dot(XTX_LCC, w_LCC) 
        ### 3. generate the secret shares of f_eval
        f_eval_SS_T = SSS_encoding(f_eval,N,T,p)
        ### 4. LCC decoding f_eval  & calculate the gradient (over the secret share)
        # 4.1. send the secret shares of f_eval
        f_deg = int(2*1 + 1)
        RT = f_deg*(K+T-1) + 1

        dec_input = np.empty((RT, d), dtype='int64')
        for j in range(1, RT+1):
            if rank == j:
                dec_input[j-1,:] = np.reshape(f_eval_SS_T[j-1,:,:], d )
                for j in range(1, N+1): # secret share q
                    if j == rank:
                        continue
                    data = np.reshape(f_eval_SS_T[j-1,:,:], d)
                    comm_volume += (d*1 * 64)
                    comm.Send(data, dest=j) # sent data to worker j
            else:
                data = np.empty(d, dtype='int64')
                comm.Recv(data, source=j)
                dec_input[j-1,:] = data # coefficients for the polynomial
        # 4.2. decode f_eval over the secret share
        dec_out = LCC_decoding(dec_input,f_deg,N,K,T, range(RT), p)
        # 4.3. compute the SS of the f function
        f_SS_T = np.zeros((d,1),dtype='int64')
        for j in range(K):
            f_SS_T = np.mod(f_SS_T + np.reshape(dec_out[j,:],(d,1)), p)
        
        ### compute the SS of the gradient
        grad_SS_T = np.mod(f_SS_T - XTy_SS_T, p)
        # truncation gradient
        grad_trunc_SS_T = MPI_TruncPr(grad_SS_T, r1_SS_T, r2_SS_T, trunc_k, trunc_scale, T, p, rank, comm, N)
        # update the model
        w_SS_T = np.mod(w_SS_T - grad_trunc_SS_T, p)
                                            
    np.savetxt("testing_results/"+str(N)+"_case/COPML_K-"+str(K)+"_T-"+str(T)+"_client-"+str(rank)+"_MNIST.txt", np.array([comm_volume]))
    

