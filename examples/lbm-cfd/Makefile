# LBM-CFD Makefile
# Supports compilation, execution, and post-processing automation

# Check for OS (Windows, Linux, Mac OS)
ifeq ($(OS),Windows_NT)
	DETECTED_OS:= Windows
else
	DETECTED_OS:= $(shell uname)
endif

# Set compiler and flags
ifeq ($(DETECTED_OS),Windows)
CXX= g++
MPIRUN= mpiexec
else
CXX= mpic++
MPIRUN= mpirun
endif
CXXFLAGS+= -std=c++14 -O3 -Wall -DOUTPUT_VORTICITY=$(OUTPUT_VORTICITY) -DOUTPUT_VELOCITY=$(OUTPUT_VELOCITY)

# Set application executable name
APP= lbmcfd

# Set source and output directories
SRCDIR= src
OBJDIR= obj
BINDIR= bin

# Simulation parameters (can be overridden)
LATTICE ?= d3q19
NPROCS ?= 1

# MPI configuration (can be overridden for different systems)
MPI_FLAGS ?=

# Output configuration parameters
OUTPUT_VORTICITY ?= 0
OUTPUT_VELOCITY ?= 0

# Set up include and library directories
ifeq ($(DETECTED_OS),Windows)
	MPI_INC= $(patsubst %\,%,$(MSMPI_INC))
	MPI_LIB= $(patsubst %\,%,$(MSMPI_LIB64))

	INC= -I"$(MPI_INC)" -I"$(MPI_INC)\x64" -I.\include
	LIB= -L"$(MPI_LIB)" -lmsmpi -lm
else
	INC= -I./include
	LIB= -lm
endif

# Link Ascent if ASCENT_DIR is set
ifneq ($(ASCENT_DIR),)
    include $(ASCENT_DIR)/share/ascent/ascent_config.mk
    ASCENT_INC_FLAGS= $(ASCENT_INCLUDE_FLAGS)
    ASCENT_LNK_FLAGS= $(ASCENT_LINK_RPATH) $(ASCENT_MPI_LIB_FLAGS)
    CXXFLAGS += -DASCENT_ENABLED
endif

# Create output directories and set output file names
ifeq ($(DETECTED_OS),Windows)
	mkobjdir:= $(shell if not exist $(OBJDIR) mkdir $(OBJDIR))
	mkbindir:= $(shell if not exist $(BINDIR) mkdir $(BINDIR))

	OBJS= $(addprefix $(OBJDIR)\, main.o)
	EXEC= $(addprefix $(BINDIR)\, $(APP).exe)
else
	mkdirs:= $(shell mkdir -p $(OBJDIR) $(BINDIR))
	
	OBJS= $(addprefix $(OBJDIR)/, main.o)
	EXEC= $(addprefix $(BINDIR)/, $(APP))
endif


# BUILD EVERYTHING
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) -o $@ $^ $(LIB) $(ASCENT_LNK_FLAGS)

ifeq ($(DETECTED_OS),Windows)
$(OBJDIR)\\%.o: $(SRCDIR)\%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INC) $(ASCENT_INC_FLAGS)
else
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INC) $(ASCENT_INC_FLAGS)
endif


# REMOVE OLD FILES
ifeq ($(DETECTED_OS),Windows)
clean:
	if exist $(OBJDIR) rmdir /s /q $(OBJDIR)
	if exist $(BINDIR) rmdir /s /q $(BINDIR)
	del /q *.vts *.pvd 2>nul || echo No VTS/PVD files to clean in root
	del /q paraview\*.vts paraview\*.pvd 2>nul || echo No VTS/PVD files to clean in paraview
else
clean:
	rm -rf $(OBJDIR) $(BINDIR)
	rm -f *.vts *.pvd
	rm -f paraview/*.vts paraview/*.pvd
	rm -f hostfile.tmp
endif

# PHONY targets
.PHONY: all clean help run pvd complete

# Help target
help:
	@echo "LBM-CFD Makefile Usage:"
	@echo ""
	@echo "  make all             - Compile the application (default)"
	@echo "  make run             - Run simulation with MPI (use in job/qsub)"
	@echo "  make pvd             - Generate PVD files for ParaView"
	@echo "  make complete        - Compile + Run + Generate PVD (use in job/qsub)"
	@echo "  make clean           - Remove compiled files and results"
	@echo ""
	@echo "Parameters:"
	@echo "  LATTICE={d3q15,d3q19,d3q27}  - Lattice model (default: $(LATTICE))"
	@echo "  NPROCS=N                     - MPI processes (default: $(NPROCS))"
	@echo "  MPI_FLAGS='flags'            - MPI runtime flags (default: none)"
	@echo "  OUTPUT_VORTICITY={0,1}       - Enable vorticity output (default: $(OUTPUT_VORTICITY))"
	@echo "  OUTPUT_VELOCITY={0,1}        - Enable velocity vector output (default: $(OUTPUT_VELOCITY))"

# Run simulation
run: $(EXEC)
	@echo "Starting LBM-CFD simulation..."
	@echo "Parameters: $(NPROCS) processes, --$(LATTICE)"
	@echo "MPI Command: $(MPIRUN) -n $(NPROCS) --hostfile hostfile.tmp $(MPI_FLAGS) $(EXEC) --$(LATTICE)"
ifeq ($(DETECTED_OS),Windows)
	$(MPIRUN) -n $(NPROCS) $(EXEC) --$(LATTICE)
else
	@echo "localhost" > hostfile.tmp
	$(MPIRUN) -n $(NPROCS) --hostfile hostfile.tmp $(MPI_FLAGS) $(EXEC) --$(LATTICE)
	@rm -f hostfile.tmp
endif
	@echo "Simulation completed."

# Generate PVD files for ParaView (only if VTS files exist)
pvd:
ifeq ($(DETECTED_OS),Windows)
	@if exist "paraview\*.vts" ( \
		echo "Generating PVD files for ParaView..." && \
		if exist "create_pvd_from_vts.py" ( \
			python "create_pvd_from_vts.py" --input-dir paraview \
		) else ( \
			echo Warning: create_pvd_from_vts.py not found \
		) && \
		echo "PVD files ready for ParaView." \
	) else ( \
		echo "No VTS files found - skipping PVD generation" \
	)
else
	@if ls paraview/*.vts >/dev/null 2>&1; then \
		echo "Generating PVD files for ParaView..."; \
		if [ -f "create_pvd_from_vts.py" ]; then \
			python3 create_pvd_from_vts.py --input-dir paraview; \
		else \
			echo "Warning: create_pvd_from_vts.py not found"; \
		fi; \
		echo "PVD files ready for ParaView."; \
	else \
		echo "No VTS files found - skipping PVD generation"; \
	fi
endif

# Complete workflow: compile + run + generate PVD
complete: all run pvd
	@echo "Complete workflow finished!"
	@echo "Ready for ParaView visualization."

