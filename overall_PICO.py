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
    print("### This is the Rank-0 Distributor ! ####")
    print("00.Load in and Process the CIFAR-10 dataset.")

    # the particular case of (K, T) pair
    K = int(np.floor((N-1)/float(3))) + 1 - int(np.floor((N-3)/float(6)))
    T = int(np.floor((N-3)/float(6)))
    print("########### For the Case : K = ", K, " and T = ", T, " #########")
    
    m = 9019*2
    d = 3073
    print("01.Data Conversion : Real to Finite Field")

    for j in range(1, N+1):
        # split the original dataset equally      
        m_j = int(m / N)
        start_row = int(m_j*(j-1))
        if j == N:
            m_j += (m%j)
        m_j = m_j - (m_j%K)
        d = 3073
        d = (N-T)*K*(int(d/((N-T)*K))+1)
        # communicate the local dataset to the correspnding client
        comm.send(m_j, dest=j) # send number of rows =  number of training samples
        comm.send(d, dest=j) # send number of columns = number of features

    print("02.Generation of all random matrices")
    comm.Barrier()

    ######################################################
    ############ for broadcasting operation in MPI #######
    ######################################################
    m_i = None
    for j in range(1, N+1):
        m_i = comm.bcast(m_i, root=j)
        data = np.empty((m_i*d), dtype="int64")
        comm.Bcast(data, root=j)
        
    for j in range(1, N+1):
        data = np.empty(int(d/K), dtype="int64")
        comm.Bcast(data, root=j) 
    
    for t in range(max_iter):
        for j in range(1, N+1):
            data = np.empty((d,1), dtype="int64")
            comm.Bcast(data, root=j)
        for j in range(1, N+1):
            data = np.empty((d,1), dtype="int64")
            comm.Bcast(data, root=j) 


######################################################
################### Clients -- rank-j ################
######################################################

elif rank <= N:

    print("### This is the client-", rank, " ####")

    # the particalr case of (K, T) pair
    K = int(np.floor((N-1)/float(3))) + 1 - int(np.floor((N-3)/float(6)))
    T = int(np.floor((N-3)/float(6)))
    
    ##################### Receive the F.F. dataset and the random mactrices #################
    # receive the raw dataset(local)
    m_i = comm.recv(source=0) # number of rows =  number of training samples
    d = comm.recv(source=0) # number of columns  = number of features
    comm.Barrier()
    
    X_i = np.random.randint(p, size=(m_i, d)) 
    y_i = np.random.randint(p, size=(m_i, 1))
    
    # receive the random matrices for encoding
    R_i = np.random.randint(p, size=(m_i, d))
    V_i = np.random.randint(p, size=(T, int(m_i/K), d))

    a_i = np.random.randint(p, size=(int(d/(N-T)), 1))
    b_i = np.random.randint(p, size=(T, int(d/((N-T)*K)), 1))
    ri_XTy = np.random.randint(p, size=(T, int(d/K), 1))

    r_i = np.random.randint(p, size=(int(d/(N-T)), 1))
    v_i = np.random.randint(p, size=(T, int(d/((N-T))), 1))

    u_i = np.random.randint(p, size=(int(d/(N-T)), 1))

    r1_trunc_i = np.random.randint(p, size=(d, 1))
    r2_trunc_i = np.random.randint(p, size=(d, 1))

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
    
    comm_volume = 0
    
    ############################################
    #         Computation of M_matrix          #
    ############################################
    M_matrix = np.empty((N,N-T), dtype="int64")
    for t in range(N):
        if t == 0:
            M_matrix[t,:] = np.ones(N-T, dtype="int64")    
        else:
            # modular operation for Overflow problem
            M_matrix[t,:] = np.mod(np.array([i**t for i in range(1, N-T+1)]), p)
    M_matrix = np.reshape(M_matrix,(N-T, N))

    ############################################
    #       Stage-1 : Dataset Encoding         #
    ############################################

    ### Offline Computation
    R_tilde = LCC_encoding_w_Random(R_i, V_i, N, K, T, p)
    # communicate R_hat to clients
    m_j_recv = []
    R_tilde_recv = []
    # the corresponding commni time -- dataset encoding
    for j in range(1, N+1):
        if rank == j:
            R_tilde_recv.append(R_tilde[j-1,:,:])
            for j_others in range(1, N+1):
                if j_others == rank: continue
                comm.send(m_i, dest=j_others)
                comm_volume += (int(m_i/K)*d * 64)
                comm.Send(np.reshape(R_tilde[j_others-1,:,:], int(m_i/K)*d), dest=j_others)
        else:
            m_j = comm.recv(source=j)
            data = np.empty(int(m_j/K)*d, dtype='int64')
            comm.Recv(data, source=j)
            comm_volume += (np.prod(data.shape) * 64)
            R_tilde_recv.append(np.reshape(data, (int(m_j/K), d)).astype('int64'))

    ### Online Computation
    X_hat = X_i - R_i
    # communicate R_hat to clients
    X_hat_recv = []
    # the corresponding communication time
    for j in range(1, N+1):
        m_j = comm.bcast(m_i, root=j)
        if rank == j:
            data = X_hat
        else:
            data = np.empty((m_j*d), dtype="int64")
        comm.Bcast(data, root=j)
        comm_volume += (np.prod(data.shape) * 64)
        m_j_recv.append(m_j)
        X_hat_recv.append(np.reshape(data, (m_j, d)))

    # Concatenation for Dataset-LCC
    #####################################
    for j in range(1, N+1):
        X_LCC_ji = np.mod(LCC_encoding_simple(X_hat_recv[j-1],N,K,T,p,rank-1) + R_tilde_recv[j-1], p)
        if j == 1:
            X_LCC_i = X_LCC_ji
        else:
            X_LCC_i = np.concatenate((X_LCC_i, X_LCC_ji), axis=0)
    #####################################

    ####### free the memory of out-of-date #####
    R_tilde = None
    R_tilde_recv = None
    X_hat = None
    X_hat_recv = None

    ############################################
    #       Stage-2 : Secret Share of XTy      #
    ############################################

    ### offline computation
    a_tilde = LCC_encoding_w_Random(a_i, b_i, N, K, T, p)
    # communicate the z_i SS to all other clients
    a_tilde_recv = np.empty((N,int(d/((N-T)*K))), dtype="int64")
    # the corresponding communication time
    for j in range(1, N+1):
        if rank == j:
            a_tilde_recv[j-1,:] = np.reshape(a_tilde[j-1,:,:], int(d/((N-T)*K)))
            for j_others in range(1, N+1):
                if j_others == rank: continue
                data = np.reshape(a_tilde[j_others-1,:,:], int(d/((N-T)*K)))
                comm_volume += (np.prod(data.shape) * 64)
                comm.Send(data, dest=j_others)
        else:
            data = np.empty(int(d/((N-T)*K)), dtype='int64')
            comm.Recv(data, source=j)
            comm_volume += (np.prod(data.shape) * 64)
            a_tilde_recv[j-1] = data
    # compute the a_tilde for client-i
    a_tilde_i = np.dot(M_matrix, a_tilde_recv)
    a_tilde_i = np.reshape(a_tilde_i, (N-T)*int(d/((N-T)*K)))
    # secret share the a_i
    a_i_SS_T = SSS_encoding(a_i,N,T,p)
    a_i_SS_recv = np.empty((N,int(d/(N-T))), dtype="int64")
    # the corresponding communication time
    for j in range(1, N+1):
        if rank == j:
            a_i_SS_recv[j-1,:] = np.reshape(a_i_SS_T[j-1,:,:], int(d/((N-T))))
            for j_others in range(1, N+1):
                if j_others == rank: continue
                data = np.reshape(a_i_SS_T[j_others-1,:,:], int(d/(N-T)))
                comm_volume += (np.prod(data.shape) * 64)
                comm.Send(data, dest=j_others)
        else:
            data = np.empty(int(d/(N-T)), dtype='int64')
            comm.Recv(data, source=j)
            comm_volume += (np.prod(data.shape) * 64)
            a_i_SS_recv[j-1] = data
    # compute the a_i tilde
    a_SS_T = np.dot(M_matrix, a_i_SS_recv)
    a_SS_T = np.reshape(a_SS_T, (N-T)*int(d/(N-T)))

    ### online computation
    yi = np.dot(X_i.T, y_i)
    # compute the Lagrange Encoding of yi
    yi_LCC = LCC_encoding_w_Random(yi, ri_XTy, N, K, T, p)
    # the corresponding communication time
    yi_LCC_recv = np.empty((N,int(d/K)), dtype="int64")
    for j in range(1, N+1):
        if rank == j:
            yi_LCC_recv[j-1,:] = np.reshape(yi_LCC[j-1,:,:], int(d/K))
            for j_others in range(1, N+1):
                if j_others == rank: continue
                data = np.reshape(yi_LCC[j_others-1,:,:], int(d/K))
                comm_volume += (np.prod(data.shape) * 64)
                comm.Send(data, dest=j_others)
        else:
            data = np.empty(int(d/K), dtype='int64')
            comm.Recv(data, source=j)
            comm_volume += (np.prod(data.shape) * 64)
            yi_LCC_recv[j-1] = data
    # compute the a_hat
    a_hat = np.sum(yi_LCC_recv, axis=0) - a_tilde_i
    # broadcasting the a_hat
    a_hat_recv = np.empty((N, int(d/K)), dtype="int64")
    for j in range(1, N+1):
        if rank == j:
            data = a_hat
        else:
            data = np.empty(int(d/K), dtype="int64")
        comm.Bcast(data, root=j)
        comm_volume += (np.prod(data.shape) * 64)
        a_hat_recv[j-1] = data
    # reconstruct for the LCC-decoded a_hat : with K terms

    RT = K + T + 1
    a_hat_decode = LCC_decoding_simple(a_hat_recv[0:K+T+1,:],N,K,T,range(RT),p)
    # compute the [XTy]_i
    XTy_SS_i = np.reshape(a_hat_decode, K*int(d/K)) + a_SS_T
    XTy_SS_i = np.reshape(XTy_SS_i, (d,1))

    a_tilde = None
    a_tilde_recv = None
    a_i_SS_T = None
    a_i_SS_recv = None
    yi = None
    yi_LCC = None
    yi_LCC_recv = None
    a_hat = None
    a_hat_recv = None
    a_hat_decode = None
    

    ############################################
    #       Stage-3 : Model Initialization     #
    ############################################

    # compute the SS of the local dataset (images + labels)
    w_init_i = (1/float(sum(m_j_recv)))*np.random.rand(int(d/(N-T)),1)
    w_init_i = my_q(w_init_i, 0, p)
    w_SS_T_i = SSS_encoding(w_init_i,N,T,p)
    # the corresponding communi volume -- whole model init
    w_SS_T = np.empty((N,int(d/(N-T))), dtype='int64')
    for j in range(1, N+1):
        if rank == j:
            w_SS_T[j-1] = np.reshape(w_SS_T_i[j-1,:,:], int(d/(N-T)))
            for j_others in range(1, N+1):
                if j_others == rank: continue
                # send the SS of the local initial model
                comm_volume += (int(d/(N-T)) * 64)
                comm.Send(np.reshape(w_SS_T_i[j_others-1,:,:], int(d/(N-T))), dest=j_others)
        else:
            data = np.empty(int(d/(N-T)), dtype='int64')
            comm.Recv(data, source=j)
            comm_volume += (np.prod(data.shape) * 64)
            w_SS_T[j-1] = data
    # compute the SS of the whole initialized model
    w_SS_T = np.dot(M_matrix, w_SS_T)
    w_SS_T = np.reshape(w_SS_T, ((N-T)*int(d/(N-T)),1))
    
    # pre-compute the X.T*X -- for poly approx of Sigmoid with r = 1
    print("before computing")
    XTX_LCC_i = X_LCC_i.T.dot(X_LCC_i)       
    print("after computing") 
    
    for iters in range(max_iter):

        ############################################
        #         Stage-4 : Model Encoding         #
        ############################################   
        
        ### Offline Computation
        # Part 01 : the LCC of random matrices
        r_i_rep = np.transpose(np.tile(np.transpose(r_i), K))            
        r_LCC = LCC_encoding_w_Random(r_i_rep,v_i,N,K,T,p)
        # communicate the r_LCC to all other clients
        r_LCC_recv = np.empty((N,int(d/(N-T))), dtype="int64")
        for j in range(1, N+1):
            if rank == j:
                r_LCC_recv[j-1,:] = np.reshape(r_LCC[j-1,:,:], int(d/(N-T)))
                for j_others in range(1, N+1):
                    if j_others == rank: continue
                    data = np.reshape(r_LCC[j_others-1,:,:], int(d/(N-T)))
                    comm_volume += (np.prod(data.shape) * 64)
                    comm.Send(data, dest=j_others)          
            else:
                data = np.empty(int(d/(N-T)), dtype='int64')
                comm.Recv(data, source=j)
                comm_volume += (np.prod(data.shape) * 64)
                r_LCC_recv[j-1,:] = data
        # compute the r_i tilde
        r_tilde_i = np.dot(M_matrix, r_LCC_recv)
        r_tilde_i = np.reshape(r_tilde_i, ((N-T)*int(d/(N-T)),1))

        # Part 02 : the SS of random matrices
        r_SS_T = SSS_encoding(r_i,N,T,p)
        # communicate the r-SS to all other clients
        r_SS_recv = np.empty((N,int(d/(N-T))), dtype="int64")
        for j in range(1, N+1):
            if rank == j:
                r_SS_recv[j-1,:] = np.reshape(r_SS_T[j-1,:,:], int(d/(N-T)))
                for j_others in range(1, N+1):
                    if j_others == rank: continue
                    data = np.reshape(r_SS_T[j_others-1,:,:], int(d/(N-T)))
                    comm_volume += (np.prod(data.shape) * 64)
                    comm.Send(data, dest=j_others)
            else:
                data = np.empty(int(d/(N-T)), dtype='int64')
                comm.Recv(data, source=j)
                comm_volume += (np.prod(data.shape) * 64)
                r_SS_recv[j-1,:] = data
        # compute the r_i tilde
        r_SS_i = np.dot(M_matrix, r_LCC_recv)
        r_SS_i = np.reshape(r_SS_i, ((N-T)*int(d/(N-T)),1))

        ### Online Computation
        W_hat_SS = w_SS_T - r_SS_i
        # communicate W_hat to clients
        W_hat_SS_recv = np.empty((N,d,1), dtype="int64")
        for j in range(1, N+1):                    
            if rank == j:
                data = W_hat_SS
            else:
                data = np.empty((d,1), dtype="int64")
            comm.Bcast(data, root=j) 
            comm_volume += (np.prod(data.shape) * 64)
            W_hat_SS_recv[j-1] = data
        # collect the W_hat_SS for SS decoding
        dec_input = np.empty((len(group_idx_set), d), dtype='int64')
        for j in group_idx_set:
            if my_worker_idx == j:
                dec_input[my_worker_idx-group_stt_idx,:] = np.reshape(W_hat_SS_recv[my_worker_idx-group_stt_idx,:,:], d)
            else:
                dec_input[j-group_stt_idx,:] = np.reshape(W_hat_SS_recv[my_worker_idx-group_stt_idx,:,:], d)
        # recover for the true model hat 
        W_hat = SSS_decoding(dec_input, group_idx_set, p) 
        W_hat = np.reshape(W_hat, (d,1)).astype('int64')

        # computation of model-LCC
        #####################################
        W_rep_hat = np.transpose(np.tile(np.transpose(W_hat), K))
        W_LCCs = LCC_encoding_simple(W_rep_hat,N,K,T,p,rank-1)
        W_LCC_i = np.mod(W_LCCs + r_tilde_i, p)
        #####################################

        ############################################
        #     Stage-5 : grad-SS & Model Update     #
        ############################################ 
        
        # Offline Computation
        f_deg = 3
        RT = f_deg*(K+T-1) + 1
        u_i = np.random.randint(p, size=(RT, int(d/(N-T)),1))
        u_i_LCC = LCC_encoding_w_Random(np.reshape(u_i[0:K,:,:], (K * int(d/(N-T)),1),), u_i[K:RT+1], N, K, RT-K, p)
        # communicate the ui-SS to all others
        u_i_LCC_recv = np.empty((N,int(d/(N-T))), dtype="int64")
        for j in range(1, N+1):
            if rank == j:
                u_i_LCC_recv[j-1,:] = np.reshape(u_i_LCC[j-1,:,:], int(d/(N-T)))
                for j_others in range(1, N+1):
                    if j_others == rank: continue
                    data = np.reshape(u_i_LCC[j_others-1,:,:], int(d/(N-T)))
                    comm_volume += (np.prod(data.shape) * 64)
                    comm.Send(data, dest=j_others)           
            else:
                data = np.empty(int(d/(N-T)), dtype="int64")
                comm.Recv(data, source=j)
                comm_volume += (np.prod(data.shape) * 64)
                u_i_LCC_recv[j-1] = data
        u_LCC = np.dot(M_matrix, u_i_LCC_recv)
        u_LCC = np.reshape(u_LCC, (d,1))
        # compute the summation of u_ik
        sum_u_ik = np.sum(u_i[0:K], axis=0)
        sum_u_ik_SS_T = SSS_encoding(sum_u_ik,N,T,p)
        # communicate the ui-SS to all others
        sum_u_ik_SS_recv = np.empty((N,int(d/(N-T))), dtype="int64")
        for j in range(1, N+1):
            if rank == j:
                sum_u_ik_SS_recv[j-1,:] = np.reshape(sum_u_ik_SS_T[j-1,:,:], int(d/(N-T)))
                for j_others in range(1, N+1):
                    if j_others == rank: continue
                    data = np.reshape(sum_u_ik_SS_T[j_others-1,:,:], int(d/(N-T)))
                    comm_volume += (np.prod(data.shape) * 64)
                    comm.Send(data, dest=j_others)           
            else:
                data = np.empty(int(d/(N-T)), dtype="int64")
                comm.Recv(data, source=j)
                comm_volume += (np.prod(data.shape) * 64)
                sum_u_ik_SS_recv[j-1] = data
        # multiple by M matrix for the sum ui_SS
        sum_ui_SS_T = np.dot(M_matrix, sum_u_ik_SS_recv)
        sum_ui_SS_T = np.reshape(sum_ui_SS_T, (d,1))
        
        # online computation
        u_i_hat = np.mod(XTX_LCC_i.dot(W_LCC_i) - u_LCC, p)
        # broadcast communication
        u_hat_recv = np.empty((N,d,1), dtype="int64")
        for j in range(1, N+1):
            if rank == j:
                data = u_i_hat
            else:
                data = np.empty((d,1), dtype="int64")
            comm.Bcast(data, root=j)
            comm_volume += (np.prod(data.shape) * 64)
            u_hat_recv[j-1,:,:] = data
        # computation of [f(X_lcc; W_lcc)]_i
        dec_input = u_hat_recv[0:RT,:,:]
        dec_input = np.reshape(dec_input, (RT,d))
        dec_out = LCC_decoding(dec_input,f_deg,N,K,T,range(RT),p)
        dec_out = np.reshape(dec_out, (K,d,1))
        # compute the overall f-SS
        dec_out = np.sum(dec_out, axis=0)
        f_SS_T = dec_out + sum_ui_SS_T

        # compute the secret share of the gradient
        grad_SS_T = np.mod(f_SS_T - XTy_SS_i, p)
        # truncate the gradient
        grad_trunc_SS_T = MPI_TruncPr(grad_SS_T,r1_trunc_i,r2_trunc_i,trunc_k,trunc_scale,T,p,rank,comm, N)
        # update the model
        w_SS_T = np.mod(np.reshape(w_SS_T, (d,1)) - grad_trunc_SS_T, p)

    np.savetxt("testing_results/"+str(N)+"_case/PICO+_K-"+str(K)+"_T-"+str(T)+"_client-"+str(rank)+"_CIFAR10.txt", np.array([comm_volume]))




