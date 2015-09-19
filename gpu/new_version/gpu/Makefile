NVCC = nvcc
MY_CC=20
NVCC_FLAGS = -fmad=true -use_fast_math -O3 -rdc=true

CU_SRCS = driver.cu reduce.cu swap.cu
CC_SRCS = Instance.cc
NVCC_CONFIG = -arch=compute_$(MY_CC) -code=sm_$(MY_CC) --ptxas-options=-v

%.o: %.cu
	$(NVCC) $(NVCC_CONFIG) $(NVCC_FLAGS) -o $@ -c $<
%.o: %.cc
	$(NVCC) $(NVCC_CONFIG) $(NVCC_FLAGS) -o $@ -c $<

MY_OBJS = $(CU_SRCS:.cu=.o)
MY_OBJS += $(CC_SRCS:.cc=.o)

tsp: $(MY_OBJS)
	$(NVCC) $(NVCC_CONFIG) $(NVCC_FLAGS) $^ -o tsp

CLEANFILES = tsp
CLEANFILES += $(MY_OBJS)
CLEANFILES += *.o[0-9][0-9]*
CLEANFILES += *.e[0-9][0-9]*
CLEANFILES += *.btr

clean:
	rm -f core *~ $(CLEANFILES)

# eof
