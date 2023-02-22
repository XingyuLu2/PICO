import numpy as np
import matplotlib.pyplot as plt
import matplotlib
N_list = [10, 20, 30, 40, 50, 60]
idx = range(len(N_list))
m = 9019*2
d = 3073

############################################################################################
############### Load in the Experimental Result files and Read #############################
############################################################################################
# the total-process time cost
total_time_max = []
total_time_mean = []
total_timeC_max = []
total_timeC_mean = []
print("### Case-2 : ####")
for N in N_list :
    K_ = [int(np.floor((N-1)/float(3)))  , int(np.floor((N-1)/float(3))) + 1 - int(np.floor((N-3)/float(6))), int(1)                       ]
    T_ = [int(1)                         , int(np.floor((N-3)/float(6)))                                    , int(np.floor((N-1)/float(3)))]
    K = K_[1]
    T = T_[1]
    print("# K = ", K, " and ", " T = ", T)
    m_ = 0
    for j in range(N):
        # split the original dataset equally      
        m_j = int(m / N)
        start_row = int(m_j*(j-1))
        if j == N:
            m_j += (m%j)
        m_ += (m_j - (m_j%K))
    print("# dataset size = ", m_)
    print("# dimension is : ", (N-T)*K*(int(d/((N-T)*K))+1))

    linearCOPML_comm_list = []
    COPML_comm_list = []
    for rank in range(1, N+1):
        comm_time = np.loadtxt("testing_results/" + str(N)+"_case/PICO+_K-"+str(K)+"_T-"+str(T)+"_client-"+str(rank)+"_CIFAR10.txt")
        linearCOPML_comm_list.append(comm_time)
        comm_time = np.loadtxt("testing_results/" + str(N)+"_case/COPML_K-"+str(K)+"_T-"+str(T)+"_client-"+str(rank)+"_CIFAR10.txt")
        COPML_comm_list.append(comm_time)
    linearCOPML_comm_list = np.array(linearCOPML_comm_list)
    COPML_comm_list = np.array(COPML_comm_list)

    ############################################################################################
    ######################## Extract out the information/record ################################
    ############################################################################################ 

    # the total-process time cost
    total_time_max.append(np.amax(linearCOPML_comm_list[:]))
    total_time_mean.append(np.mean(linearCOPML_comm_list[:]))
    total_timeC_max.append(np.amax(COPML_comm_list[:]))
    total_timeC_mean.append(np.mean(COPML_comm_list[:]))
    

total_time_max = np.array(total_time_max)
total_time_mean = np.array(total_time_mean)
total_timeC_max = np.array(total_timeC_max)
total_timeC_mean = np.array(total_timeC_mean)

np.savetxt("online_COPML_CIFAR10.txt", total_timeC_max)
np.savetxt("online_PICO_CIFAR10.txt", total_time_max)

print("############# Performance gain: Online Training, Case-2 ################")
gain_list = []
for j in range(6):
    gain_list.append(total_timeC_max[j]/total_time_max[j])
    print("For N=", N_list[j], ", the gain is : ", total_timeC_max[j]/total_time_max[j])

### plotting the experimental records
fig, ax = plt.subplots(figsize=(8,6),dpi=500)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color("black")
    ax.spines[axis].set_zorder(0)
plt.figure(1)
plt.plot(idx, total_time_max,color='r',marker='v')
plt.plot(idx, total_timeC_max,color='b',marker='X')
# plt.title("Comparison of Online Training Time-Cost -- CIFAR10, 40BW, with Group")
plt.xlabel('N (number of clients)', fontweight='bold', fontsize=18)
plt.ylabel('Time (sec)', fontweight='bold', fontsize=18)
plt.legend(['PICO', 'COPML'], prop={'size': 18})
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
plt.xticks(idx, N_list)
plt.annotate(s='', xy=(5,total_time_max[5]), xytext=(5,total_timeC_max[5]), arrowprops=dict(arrowstyle='<->'),fontsize=18)
plt.text(4.5,total_timeC_max[5]-1500, r'6.8X',fontsize=18)
plt.grid()
plt.savefig('online_training_CIFAR10.png')

plt.show()
