# LBM-CFD

**Lattice-Boltzmann Method 3D Computational Fluid Dynamics Simulation**

3D LBM simulation parallelized using MPI with ParaView visualization support

## Simulation Time Stepping

Unlike fixed time step simulations, this LBM-CFD implementation uses **adaptive time stepping** based on stability conditions:

- **CFL Condition**: Ensures fluid doesn't move more than one grid cell per time step
- **Diffusion Stability**: Prevents numerical instabilities in viscous flows  
- **Automatic Selection**: The simulation automatically chooses the smaller (more restrictive) time step

### Key Implications:
- **Total simulation time varies** based on grid resolution and flow parameters
- **Higher resolution** = smaller time steps = longer simulation time  
- **Faster flows** = smaller time steps = more iterations needed
- The simulation reports the calculated time step and total simulated time at startup

## Prerequisites

- **C++ Compiler**: g++ (Windows) or mpic++ (Linux/Unix/Mac OS)
- **MPI**: OpenMPI/MPICH
- **Python**: For ParaView file generation (optional)
- **ParaView**: For visualization (optional)

## Configuration Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `LATTICE` | `d3q15`, `d3q19`, `d3q27` | `d3q19` | Lattice model |
| `N` | Integer | `1` | Number of MPI processes |
| `PPN` | Integer | Auto | Processes per node |
| `OUTPUT_VELOCITY` | `0`, `1` | `0` | Enable velocity output |
| `OUTPUT_VORTICITY` | `0`, `1` | `0` | Enable vorticity output |

## Makefile Targets

- `make all` - Compile application (default)
- `make run` - Run simulation
- `make pvd` - Generate ParaView files
- `make complete` - Build + Run + Generate PVD
- `make clean` - Remove compiled files

## Building and Running
```
cd examples/lbm-cfd
make 
make run N=<num_procs> PPN=<processes_per_node> LATTICE=<lattice_model>
```

### Complete Workflow with ParaView Output Files
```
make complete N=<num_procs> PPN=<processes_per_node> LATTICE=<lattice_model> OUTPUT_VELOCITY=1 OUTPUT_VORTICITY=1
```
