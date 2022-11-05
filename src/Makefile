CUDA_ROOT =/usr/cuda-11.8
MPI_ROOT=/opt/share/openmpi-3.0.0/no_cuda

CUDA_CPP=$(CUDA_ROOT)/bin/nvcc
CPP=gcc
LD=$(MPI_ROOT)/bin/mpicxx

CXXFLAGS = -O3

LDFLAGS = -L$(CUDA_ROOT)/lib64 -lcudart -L$(CUDA_ROOT)/lib64/stubs -lcuda

CXXFLAGS+= \
           -I$(MPI_ROOT)/include \
           -I$(CUDA_ROOT)/include \
           -I$(CUDA_ROOT)/samples/common/inc

CUDA_CXXFLAGS = $(CXXFLAGS)

SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

CUDA_SRC=$(wildcard *.cu)
CUDA_OBJ=$(CUDA_SRC:%.cu=%.o)

NMR_exe: $(OBJ) $(CUDA_OBJ)
	$(LD) -o $@ $^ $(LDFLAGS)

%.o : %.cu
	$(CUDA_CPP) $(CUDA_CXXFLAGS) -c $< -o $*.o

%.o : %.cpp
	$(CPP) $(CXXFLAGS) -c $< -o $*.o
clean:
	rm -rf *.o NMR_exe
