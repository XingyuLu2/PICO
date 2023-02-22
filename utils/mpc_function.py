from mpi4py import MPI
import numpy as np
import random
from array import array
import math
import time

np.random.seed(42) # set the seed of the random number generator for consistency

# p = 15485863 # field size

def modular_inv(a,p):
    x, y, m = 1, 0, p
    while a > 1:
        q = a//m;
        t = m ;

        m = np.mod(a,m)
        a = t
        t = y

        y, x = x - np.int64(q)*np.int64(y), t

        if x<0:
            x = np.mod(x,p)
    return np.mod(x,p)

def divmod(_num, _den, _p):
    # compute num / den modulo prime p
    _num = np.mod(_num,_p)
    _den = np.mod(_den,_p)
    _inv = modular_inv(_den,_p)
    # print(_num,_den,_inv)
    return np.mod(np.int64(_num) * np.int64(_inv), _p)

def PI(vals,p):  # upper-case PI -- product of inputs
    accum = 1
    for v in vals:
        accum = np.mod(accum*v,p)
    return accum

def gen_Lagrange_coeffs(alpha_s,beta_s,p,is_K1=0):
    if is_K1==1:
        num_alpha = 1
    else:
        num_alpha = len(alpha_s)
    U = np.zeros((num_alpha, len(beta_s)),dtype='int64')
#         U = [[0 for col in range(len(beta_s))] for row in range(len(alpha_s))]
    #print(alpha_s)
    #print(beta_s)
    for i in range(num_alpha):
        for j in range(len(beta_s)):
            cur_beta = beta_s[j]

            den = PI([cur_beta - o   for o in beta_s if cur_beta != o], p)
            num = PI([alpha_s[i] - o for o in beta_s if cur_beta != o], p)
            U[i][j] = divmod(num,den,p)
            # for debugging
            # print(i,j,cur_beta,alpha_s[i])
            # print(test)
            # print(den,num) 
    return U.astype('int64')

def SSS_encoding(X,N,T,p):
    m = len(X)
    d = len(X[0])

    alpha_s = range(1,N+1)
    alpha_s = np.int64(np.mod(alpha_s,p))
    X_BGW = np.zeros((N,m,d),dtype='int64')
    R = np.random.randint(p,size=(T+1,m,d))
    R[0,:,:] = np.mod(X, p)

    for i in range(N):
        for t in range(T+1):
            X_BGW[i,:,:] = np.mod(X_BGW[i,:,:] + R[t,:,:]*(alpha_s[i]**t), p)

    return X_BGW

def gen_BGW_lambda_s(alpha_s, p):
    lambda_s = np.zeros((1, len(alpha_s)),dtype='int64')

    for i in range(len(alpha_s)):
            
        cur_alpha = alpha_s[i];

        den = PI([cur_alpha - o   for o in alpha_s if cur_alpha != o], p)
        num = PI([0 - o for o in alpha_s if cur_alpha != o], p)
        lambda_s[0][i] = divmod(num,den,p)
    return lambda_s.astype('int64')

def SSS_decoding(f_eval,worker_idx,p): # decode the output from T+1 evaluation points
    # f_eval     : [RT X d ]
    # worker_idx : [ 1 X RT]
    # output     : [ 1 X d ]

    #t0 = time.time()
    max_id = np.max(worker_idx) + 2
    alpha_s = range(1,max_id)
    alpha_s = np.mod(alpha_s,p).astype("int64")
    alpha_s_eval = [alpha_s[i] for i in worker_idx]
    #t1 = time.time()
    # print(alpha_s_eval)
    lambda_s = gen_BGW_lambda_s(alpha_s_eval, p).astype('int64')
    #t2 = time.time()
    # print(lambda_s.shape)
    f_recon = np.mod(np.dot(lambda_s,f_eval), p)
    #t3 = time.time()
    #print 'time info for BGW_dec', t1-t0, t2-t1, t3-t2
    return f_recon


def LCC_encoding(X,N,K,T,p):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K+T,m//K,d),dtype='int64')
    for i in range(K):
        X_sub[i] = X[i*m//K:(i+1)*m//K:]
    for i in range(K,K+T):
        X_sub[i] = np.random.randint(p, size=(m//K,d))

    n_beta = K+T
    stt_b, stt_a = -int(np.floor(n_beta/2)), -int(np.floor(N/2))
    beta_s, alpha_s = range(stt_b,stt_b+n_beta), range(stt_a,stt_a+N)
    alpha_s = np.array(np.mod(alpha_s,p)).astype('int64')
    beta_s = np.array(np.mod(beta_s,p)).astype('int64')

    U = gen_Lagrange_coeffs(alpha_s,beta_s,p)
    # print U

    X_LCC = np.zeros((N,m//K,d),dtype='int64')
    for i in range(N):
        for j in range(K+T):
            X_LCC[i,:,:] = np.mod(X_LCC[i,:,:] + np.mod(U[i][j]*X_sub[j,:,:],p),p)
    return X_LCC

def LCC_encoding_w_Random(X,R_,N,K,T,p):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K+T,m//K,d),dtype='int64')
    for i in range(K):
        X_sub[i] = X[i*m//K:(i+1)*m//K:]
    for i in range(K,K+T):
        X_sub[i] = R_[i-K,:,:].astype('int64')

    n_beta = K+T
    stt_b, stt_a = -int(np.floor(n_beta/2)), -int(np.floor(N/2))
    beta_s, alpha_s = range(stt_b,stt_b+n_beta), range(stt_a,stt_a+N)
    alpha_s = np.array(np.mod(alpha_s,p)).astype('int64')
    beta_s = np.array(np.mod(beta_s,p)).astype('int64')

    U = gen_Lagrange_coeffs(alpha_s,beta_s,p)
    # print U

    X_LCC = np.zeros((N,m//K,d),dtype='int64')
    for i in range(N):
        for j in range(K+T):
            X_LCC[i,:,:] = np.mod(X_LCC[i,:,:] + np.mod(U[i][j]*X_sub[j,:,:],p),p)
    return X_LCC

def LCC_encoding_simple(X,N,K,T,p,dest_j):
    m = len(X)
    d = len(X[0])
    X_sub = np.zeros((K,m//K,d),dtype='int64')
    for i in range(K):
        X_sub[i] = X[i*m//K:(i+1)*m//K:]

    n_beta = K+T
    stt_b, stt_a = -int(np.floor(n_beta/2)), -int(np.floor(N/2))
    beta_s, alpha_s = range(stt_b,stt_b+n_beta), range(stt_a,stt_a+N)
    alpha_s = np.array(np.mod(alpha_s,p)).astype('int64')
    beta_s = np.array(np.mod(beta_s,p)).astype('int64')

    U = np.zeros(len(beta_s),dtype='int64')
    for j in range(0, K): # the beta-k in the denomi term
        cur_beta = beta_s[j]
        den = PI([cur_beta - o   for o in beta_s if cur_beta != o], p)
        num = PI([alpha_s[dest_j] - o for o in beta_s if cur_beta != o], p)
        U[j] = divmod(num,den,p)
    U = U.astype('int64')

    X_LCC = np.zeros((m//K,d),dtype='int64')
    for j in range(K):
        X_LCC[:,:] = np.mod(X_LCC[:,:] + np.mod(U[j]*X_sub[j,:,:],p),p)
    return X_LCC

def LCC_encoding_w_Random_partial(X,R_,N,K,T,p,worker_idx):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K+T,m//K,d),dtype='int64')
    for i in range(K):
        X_sub[i] = X[i*m//K:(i+1)*m//K:]
    for i in range(K,K+T):
        X_sub[i] = R_[i-K,:,:].astype('int64')

    n_beta = K+T
    stt_b, stt_a = -int(np.floor(n_beta/2)), -int(np.floor(N/2))
    beta_s, alpha_s = range(stt_b,stt_b+n_beta), range(stt_a,stt_a+N)
    alpha_s = np.array(np.mod(alpha_s,p)).astype('int64')
    beta_s = np.array(np.mod(beta_s,p)).astype('int64')
    alpha_s_eval = [alpha_s[i] for i in worker_idx]

    U = gen_Lagrange_coeffs(alpha_s_eval,beta_s,p)
    # print U

    N_out = U.shape[0]
    X_LCC = np.zeros((N_out,m//K,d),dtype='int64')
    for i in range(N_out):
        for j in range(K+T):
            X_LCC[i,:,:] = np.mod(X_LCC[i,:,:] + np.mod(U[i][j]*X_sub[j,:,:],p),p)
    return X_LCC


def LCC_decoding(f_eval,f_deg,N,K,T,worker_idx,p):
    RT_LCC = f_deg*(K+T-1) + 1

    n_beta = K #+T
    stt_b, stt_a = -int(np.floor(n_beta/2)), -int(np.floor(N/2))
    beta_s, alpha_s = range(stt_b,stt_b+n_beta), range(stt_a,stt_a+N)
    alpha_s = np.array(np.mod(alpha_s,p)).astype('int64')
    beta_s = np.array(np.mod(beta_s,p)).astype('int64')
    alpha_s_eval = [alpha_s[i] for i in worker_idx]
    
    U_dec = gen_Lagrange_coeffs(beta_s,alpha_s_eval,p)

    # print U_dec 

    f_recon = np.mod((U_dec).dot(f_eval),p)

    return f_recon.astype('int64')

############################################################################
def LCC_decoding_simple(f_eval,N,K,T,worker_idx,p):

    n_beta = K
    stt_b, stt_a = -int(np.floor(n_beta/2)), -int(np.floor(N/2))
    beta_s, alpha_s = range(stt_b,stt_b+n_beta), range(stt_a,stt_a+N)
    alpha_s = np.array(np.mod(alpha_s,p)).astype('int64')
    beta_s = np.array(np.mod(beta_s,p)).astype('int64')
    alpha_s_eval = [alpha_s[i] for i in worker_idx]
    
    U_dec = gen_Lagrange_coeffs(beta_s,alpha_s_eval,p)

    f_recon = np.mod((U_dec).dot(f_eval),p)

    return f_recon.astype('int64')

############################################################################


def my_q(X,q_bit,p):
    X_int = np.round(X*(2**q_bit))
    is_negative = (abs(np.sign(X_int)) - np.sign(X_int))/2
    out = X_int + p * is_negative
    return out.astype('int64')

def my_q_inv(X_q,q_bit,p):
    flag = X_q - (p-1)/2
    is_negative = (abs(np.sign(flag)) + np.sign(flag))/2
    X_q = X_q - p * is_negative
    return X_q.astype(float)/(2**q_bit)


def MPI_TruncPr(in_SS_T, r1_SS_T, r2_SS_T, trunc_k, trunc_m, T, p, rank, comm, N):
    t0 = time.time()
    
    a_SS_T = in_SS_T.astype('int64')
    trunc_size = np.prod(a_SS_T.shape)

    a_SS_T = np.reshape(a_SS_T,trunc_size)
    r1_SS_T = np.reshape(r1_SS_T,trunc_size)
    r2_SS_T = np.reshape(r2_SS_T,trunc_size)    

    t1 = time.time() 

    b_SS_T = np.mod(a_SS_T + 2**(trunc_k-1), p)

    r_SS_T = np.mod((2**trunc_m)*r2_SS_T + r1_SS_T , p)

    c_SS_T = np.mod( b_SS_T + r_SS_T ,p)
    # print 'rank=',rank, c_SS_T.shape
    
    t2 = time.time() 

    dec_input = np.empty((T+1, trunc_size), dtype='int64')
    for j in range(1, T+2):
        if rank == j:
            dec_input[j-1,:] = c_SS_T
            for j in range(1, N+1): # secret share q
                if j == rank :
                    continue
                data = c_SS_T
                comm.Send(data, dest=j) # sent data to worker j
        else:
            data = np.empty(trunc_size, dtype='int64')
            comm.Recv(data, source=j)
            dec_input[j-1,:] = data # coefficients for the polynomial

    t3 = time.time() 

    c_dec = SSS_decoding(dec_input, range(T+1), p) 
    # print 'rank=',rank, 'c_dec is completed', c_dec.shape

    t4 = time.time()
    c_prime = np.mod( np.reshape(c_dec, trunc_size), 2**trunc_m )

    a_prime_SS_T = np.mod(c_prime - r1_SS_T, p)

    d_SS_T = np.mod(a_SS_T- a_prime_SS_T, p)
    
    t5 = time.time() 
    d_SS_T = divmod(d_SS_T, 2**trunc_m, p)

    d_SS_T = np.reshape(d_SS_T, in_SS_T.shape)

    t6 = time.time() 
    #time_set = np.array([t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5])
    #print('time info for trunc pr',time_set)
    return d_SS_T.astype('int64')
