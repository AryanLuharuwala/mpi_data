CXX = mpicxx
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -g -fopenmp
NVCCFLAGS = -std=c++14 -g -Xcompiler -fopenmp
LDFLAGS = -lcudart

OBJS = code.o master.o worker.o dataUtils.o cuda.o

all: mpi_dem

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

cuda.o: cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

mpi_dem: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f *.o mpi_dem
