#!/bin/bash
let num=1
for N in 10 20 30 40 50 60
do
   mpiexec -n $(($N+$num)) python3 overall_PICO.py $N
done

