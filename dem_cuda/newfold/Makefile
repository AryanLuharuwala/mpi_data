CXX = mpicxx
CXXFLAGS = -std=c++17 -Wall -g -fopenmp

OBJS = code.o master.o worker.o dataUtils.o

all: mpi_dem

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

mpi_dem: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

clean:
	rm -f *.o mpi_dem
