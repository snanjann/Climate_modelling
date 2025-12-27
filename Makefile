CXX ?= g++
SRC := main.cpp \
	src/utils/cli.cpp \
	src/utils/geodata.cpp \
	src/utils/ghosts.cpp \
	src/utils/io.cpp \
	src/utils/physics.cpp \
	src/utils/stats.cpp \
	src/utils/weather.cpp \
	src/models/heat_models.cpp

# Detect a usable OpenMP toolchain (clang on macOS needs extra flags).
UNAME_S := $(shell uname -s)
OPENMP_FLAGS :=
OPENMP_LIBS :=
ifeq ($(UNAME_S),Darwin)
    OPENMP_FLAGS := -Xpreprocessor -fopenmp
    OPENMP_LIBS := -lomp
    OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    ifneq ($(OMP_PREFIX),)
        OPENMP_FLAGS += -I$(OMP_PREFIX)/include
        OPENMP_LIBS += -L$(OMP_PREFIX)/lib
    endif
else
    OPENMP_FLAGS := -fopenmp
endif

CXXFLAGS ?= -O3 -std=c++17
CXXFLAGS += -std=c++17 -Isrc $(OPENMP_FLAGS)
LDFLAGS += $(OPENMP_FLAGS) $(OPENMP_LIBS)

solver: $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f solver
