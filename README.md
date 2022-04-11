# Lake - by Peilin Rao (peilinr) and Zheng Zhong (zhengzho)

## Summary
We are going to implement a dynamic water surface simulator that can be accelerated by NVIDIA GPUs. 
The simulation focuses on providing real dispersion properties of water waves and shows the behavior of water surface when interacts with objects.

## Background
The application we are going to implement is a simulation of the water surface that is constantly being disturbed.
The fluid simulation is actually a multi-layered computation. The most common way of doing this is to adopt pyramid generation kernels. 
We will implement a multiresolution dispersion kernel with low-frequency compensation that supports interaction with objects and shadow mask propagation. The multilayer particle-wise upsampling and downsampling involved in implementing this multiresolution dispersion kernel can be benefit from parallel programming. A CUDA based solution breaks the water surface into different blocks, performs shared-varibale computation within blocks and synchronize between blocks after each layer of computation.

## Challenge
Although we've did some research in the implementation of water wave simulation, the effect of this multiresolution dispersion kernel mechanism remains unclear for us. The determination of where, when and how to do synchronization between blocks is challenging. Besides, finding the balance between computational cost and simulation accuracy in this implementation can be hard.  We hope to learn good methods and give some improvement on parallizing computer graphics code in this project.

## Workload
There is a high communication to computation ratio in this pyramid generation kernels. Also the layers of adjustments (for example a rock is thrown into the lake, or some wind approaches the water surface, causing different particle behavior of the water surface) should be able to access with good locality and communicate between thread/blocks with the minimum information transfering.

## Constraints
The randomness of the disturbing position of the water surface can easily create workload imbalancing in parallization this computation. We need to find a good way to balance the workload per thread when different extents disturbing happens at arbitrary position.

## Resources
Our inspiration of this project comes from [this paper](http://www.gmrv.es/Publications/2016/CMTKPO16/main.pdf), which ellaborate on the methods of designing a " compact multiresolution dispersion kernel" for fluid simulation that is based on pyramid kernels. 

## Goal and Deliverables

HOPE TO ACHIEVE ("75%"): fully functional parallelized water wave simulation without features such as reflecting boundary (wave reflects when touching the shore) and shadow mask propogation (a method that improves the efficient shared memory usage).  

PLAN TO ACHIEVE ("100%"): fully functional parallelized water wave simulation with features such as reflecting boundary and shadow mask propogation.  

EXTRA GOAL ("125")： improve the algorithm of multiresolution dispersion kernel with either better resolution/visual effects or better performance. 

Our poster session's demo would be interactive. Other students can see our lake simulation dynamically changing when they "throw" an object into the water. They can directly see the speedup by parallislem during any time period or over any action taken.  

## Platform Choice
We plan to use C++ with CUDA support for implementation and OpenGL for persenting the simulating results. We choose to use CUDA since NVIDIA GPUs are the most common way to accelerate computer graphics algorithms. OpenGL is powerful enough to draw the waves in order to present our beautiful simulated lake.

## Schedule
1st week: read paper, do more research about this problem, and plan our implementation  
2nd week: implement the sequential version and try to get the algorithm working  
3rd week: implement the naive parallel version, benchmark its performance. (Submit it for intermediate checkpoint)  
4th week: improve the parallel version of the program  
5th week: perform analysis and write report  


## Milestone
As the milestone, we finished research about wave equations and algorithms needed for the sequentail version of the lake simulator. We have a demo with python and matplotlib about the fluid propagation.

We still believe that we are able to produce all the deliverables. We will show a demo about the sequential and GPU accelerated version of lake simulations. We have no preliminary results since we have not started to build the GPU version of it. The main issue that concerns me is the package choice for animation representation since I think matplotlib might not have good GPU support. If it really turns out to be the case, we will switch to another graphics package that has generic GPU support.
