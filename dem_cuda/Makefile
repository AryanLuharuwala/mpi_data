# Auto-detect nvcc and CUDA install prefix
NVCC ?= $(shell which nvcc)
CUDA_PATH ?= $(shell dirname $(shell dirname $(NVCC)))

# Auto-detect FAISS_PATH via LD_LIBRARY_PATH; override with env if needed
FAISS_PATH ?= $(shell \
  for d in $(subst :, ,$(LD_LIBRARY_PATH)); do \
    if [ -f $$d/libfaiss.so ]; then echo $${d%/lib}; break; fi; \
  done)

# Directories
SRC_DIR = newfold/src
INC_DIR = newfold/include
OBJ_DIR = newfold/obj
BIN_DIR = newfold/bin

# Include paths
INCLUDES = -I$(CUDA_PATH)/include -I$(INC_DIR)

# Enable Intel compiler if USE_INTEL=1, otherwise default MPI C++ compiler
USE_INTEL ?= 0
ifeq ($(USE_INTEL),1)
  CXX := mpiicpc
  CPP_OMP_FLAG := -qopenmp
  NVCC_OMP_FLAG := -Xcompiler -qopenmp
  NVCC_CCFLAG := -ccbin $(CXX)
else
  CXX := mpicxx
  CPP_OMP_FLAG := -fopenmp
  NVCC_OMP_FLAG := -Xcompiler -fopenmp
  NVCC_CCFLAG :=
endif

# Compiler flags
CXXFLAGS := -std=c++17 -Wall -g $(CPP_OMP_FLAG) $(INCLUDES)
NVCCFLAGS := -std=c++17 -g $(NVCC_OMP_FLAG) $(NVCC_CCFLAG) $(INCLUDES)

# Linker flags
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(FAISS_PATH)/lib -lcudart -lfaiss $(CPP_OMP_FLAG)

# Source files - exclude cuda_check.cpp which has its own main function
CPP_SRCS = $(filter-out $(SRC_DIR)/cuda_check.cpp,$(wildcard $(SRC_DIR)/*.cpp))
CUDA_SRCS = $(wildcard $(SRC_DIR)/*.cu)
HEADERS = $(wildcard $(INC_DIR)/*.h) $(wildcard $(INC_DIR)/*.cuh)

# Object files - convert path from src/file.cpp to obj/file.o
CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CUDA_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CUDA_SRCS))
OBJS = $(CPP_OBJS) $(CUDA_OBJS)

# Target
TARGET = $(BIN_DIR)/mpi_dem

# Phony targets
.PHONY: all clean depend

# Default target
all: $(TARGET)

# Create directories if they don't exist
$(shell mkdir -p $(BIN_DIR) $(OBJ_DIR))

# Compile C++ source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link the executable
$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

# Clean up
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)
	@echo "Cleaned up build files"

# Generate dependencies
depend:
	makedepend -- $(CXXFLAGS) -- $(CPP_SRCS)
	makedepend -- $(NVCCFLAGS) -- $(CUDA_SRCS)