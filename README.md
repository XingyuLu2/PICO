# PICO

This contains the implementation of the PICO algorithm.

Download it and run it as Python programs. Our experiment is based on two diferent datasets:
1. CIFAR10: 
https://www.cs.toronto.edu/~kriz/cifar.html
2. MNIST: 
http://yann.lecun.com/exdb/mnist/

How to run the program:
1. to run the PICO algorithm and get the communication volume with different number of clients [10, 20, 30, 40, 50, 60], compile the "run_PICO.sh" and run as:
'''
chmod +x run_PICO.sh
./run_PICO.sh
'''

1. to run the COPML as benchmark and get the communication volume with different number of clients [10, 20, 30, 40, 50, 60], compile the "run_PICO.sh" and run as:
'''
chmod +x run_PCOPML.sh
./run_COPML.sh
'''
