# Project Lake - by Peilin Rao (peilinr) and Zheng Zhong (zhengzho)

## Execution

To run the code use:  

    mpiexec -n [k] python -m mpi4py .\lake_basic.py [method]
or 

    python -m mpi4py .\lake_basic.py [method]

First command can run any method but the second command cannot run method 4 or 5.  
k: any power of two value  
method:  
(1) BF: Brute Force without any vectorization  
(2) DFT: Vectorized Discrete Fourier Transform  
(3) FFT: Vectorized Fast Fourier Transform  
(4) FFT_P: Vectorized MPI-Parallel Fast Fourier Transform  
(5) FFT_PC: Vectorized MPI-Parallel Fast Fourier Transform with Cuda Acceleration  

## Final Report
https://github.com/peilinrao/parallel-computing-project-lake/blob/main/Final%20Report.pdf

## Presentation

https://drive.google.com/file/d/1cl3npeVgucno95QVSl2dOtneOwK4Ccgh/view?usp=sharing

## Summary
In this project, we use a method of simulating water surface with statistical model.
Using statistical model gives more realistic result but it is much more computational expensive than
the approximation model, which tends to oversimplify the water surface to a linear combination of
sin waves. After we implement this statistical model with the brute force algorithm, we modify it
to get a good vectorization version of SIMD optimization. Then, we find out the calculation in the
model can be decomposed into two Fourier transforms, which can be accelerated with Fast Fourier
Transform algorithm. We implement a parallel version of Fast Fourier Transfer with Message Passing
Interface. Finally, we add CUDA supports in our simulation which greatly reduces the computation
time. With MPI and CUDA, our implementation achieves a significant speedup compared to the
sequential version. We would like to elaborate our implementation and discussion experimental
results on different inputs. 

## Background
We would like to begin by giving some characteristics and constraints considered in the statistical
model before talking about its implementation. We would not focus on any rendering problem such as lighting or
coloring since this is not a project for computer graphics course. Instead, we only consider the height
(y) change of a tile of points in 3D dimension with fixed x and z values. The output of our program is
the triangular mesh of those points moving over time. We would also use paralleized FFT with MPI for 
acceleration.

## Challenge
Water wave simulation is naturally a hard problem. The math and physics behind is complex. Even after
we implement this model we need to find good way of utilizing parallelization. Ideas such as MPI or 
CUDA implementation requires rewrites of sequential code and introduce some constaints on mapping cores
or threads to jobs. The computation complexity lays under two places: the physics simulation workload and
the workload introduced by parallelization. We hope to learn good methods and give some improvement on
parallizing computer graphics code in this project.

## Resources
Jerry Tessendorf. Simulating ocean water. pp. 26, 2004.

## Goal and Deliverables

HOPE TO ACHIEVE ("75%"): fully functional vectorized water wave simulation

PLAN TO ACHIEVE ("100%"): fully functional vectorized water wave simulation with MPI-Parallel FFT

EXTRA GOAL ("125")： fully functional vectorized water wave simulation with MPI-Parallel FFT and CUDA Acceleration

Our poster session's demo would be interactive. Others can play with our code to adjust wind velocity, wind direction, methods or sample sizes. 

## Platform Choice
We plan to use C++ with CUDA support for implementation and MPI for accelerating FFT. We choose to use CUDA since NVIDIA GPUs are the most common way to accelerate computer graphics algorithms.

## Schedule
3.23 - 4.11[Done]: read the paper and plan for our implementation <br />
4.11 - 4.20: implement the sequential version <br />
4.20 - 4.24: implement the naive parallel version, benchmark its performance. (Submit it for intermediate checkpoint) <br />
4.24 - 4.29: improve the parallel version of the program and draft the report with previous benchmarks <br />
4.29 - 5.5: prepare for the poster session 


## Milestone
As planned previously, we finished researching about [wave equations and algorithms](http://www.coastalwiki.org/wiki/Shallow-water_wave_theory#Derivation_of_the_Airy_Wave_equations) needed for the sequential version of our lake simulator. We have a demo with python and matplotlib about the fluid propagation.

We still believe that we are able to produce all the deliverables. We plan to show a demo about the sequential and GPU accelerated version of lake simulations at the poster sessino. We have no preliminary results since we have not started to build the GPU version of it. The main concern we have right now is the package choice for animation representation since matplotlib might not have good GPU support. If this turns out to be true, we will try to find another graphics package that has generic GPU support. We plan to use numba for GPU computation acceleration on python programs. The revised timeline is also updated above in the schedule section.
