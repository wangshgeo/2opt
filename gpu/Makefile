#CUDA_PATH = /usr/local/cuda
CUDA_PATH = /usr/local/cuda-10.0
MYARCH = 20
CC = g++ 
NVCC = $(CUDA_PATH)/bin/nvcc
CFLAGS = -L$(CUDA_PATH)/lib64 -Wl,-rpath -Wl,$(CUDA_PATH)/lib64 -lcudart -std=c++0x -O3
NVCCFLAGS= -O3 -arch=compute_$(MYARCH) -code=sm_$(MYARCH) -I$(CUDA_SDK_PATH)/C/common/inc --ptxas-options=-v -fmad=true -use_fast_math

opt2_CUSRCS = opt2.cu
opt2_CSRCS = driver.c timer.c solution.cpp instance.cpp opt2.cpp
opt2_CUOBJS = $(opt2_CUSRCS:.cu=.o__cu)
opt2_COBJS = $(opt2_CSRCS:.c=.o__c)

opt2: $(opt2_CUOBJS) $(opt2_COBJS)
	$(CC) $(CFLAGS) $^ -o $@ 


%.o__c: %.c
	$(CC) -o $@ -c $<

%.o__cu: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

clean:
	rm -f core *.o__cu *.o__c *~ opt2

# eof
