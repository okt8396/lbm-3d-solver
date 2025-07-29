# Ascent-Trame (2D)
Bridge for accessing Ascent extracts in a Trame application


### MPI
Install desired version of MPI (e.g. OpenMPI, MPICH, etc.)


### Python Virtual Environment
```
cd lbm-cfd
python -m venv ./.venv
source .venv/bin/activate
pip install setuptools
pip install numpy
pip install opencv-python
pip install trame
pip install trame-vuetify
pip install trame-rca
pip install --no-binary :all: --compile mpi4py
```


### Ascent Install and Build (with MPI)
```
git clone --recursive https://github.com/alpine-dav/ascent.git
cd ascent
```

Edit "./scripts/build_ascent/build_ascent.sh" to change mfem version from 4.6 to 4.7

Make sure that the Python virtual environment created in the previous step is activated

```
env enable_python=ON enable_mpi=ON prefix=<ascent_install_dir> ./scripts/build_ascent/build_ascent.sh
```

# Paraview (3D)

## Visualization with ParaView

The simulation generates VTS files containing vorticity and/or velocity data based on user-defined arguments. The files can be found in the `paraview/` directory. Use `make pvd` after running the simulation to create PVD files

There are two options to perform the visualization:
### Option #1: Local Visualization

To visualize the results, follow these steps:
1. Download the generated VTS/PVD files
2. Launch ParaView locally
3. In ParaView, go to File > Open and select  VTS/PVD files
4. Apply filters and settings to visualize the data

### Option #2: Remote Visualization using ParaView Modules*
Copy the name of the assigned compute node, and run the following in a new terminal:
```
ssh -L 11111:<node_name>:11111 <username>@<hostname>.alcf.anl.gov
```
Back on the compute node, start the ParaView server:
```
module use /soft/modulefiles
module load visualization/paraview/paraview-5.13.3
pvserver --server-port=11111
```
Then connect to localhost:11111 in ParaView's "Connect" dialog box

#


*Make sure locally installed version of ParaView matches the version used on the compute node

*More detailed instructions for the remote ParaView connection can be found here: https://docs.alcf.anl.gov/polaris/visualization/paraview-manual-launch/#setting-up-paraview
