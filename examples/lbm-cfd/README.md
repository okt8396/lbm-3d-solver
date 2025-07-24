# Visualization

The simulation generates VTS files containing either vorticity or velocity data depending on the build flags. The files can be found in the `paraview/` directory. Use `make pvd` after `make run` to create ParaView-compatible PVD files, then load them in ParaView.

There are two options to do this visualization:
## Option #1: Local Visualization
download the files to your local machine and open them directly in ParaView
```
scp -r <username>@<hostname>.alcf.anl.gov://<path_to_remote_paraview_directory> <path_to_local_directory>
```

## Option #2: Remote Visualization using Sophia's ParaView Modules
run the following on a Sophia compute node:
```
ip route get 8.8.8.8
```
copy the IP address from the above command, and run the following in a new terminal:
```
ssh -L 11111:<remote_ip>:11111 <username>@sophia.alcf.anl.gov
```
back on the Sophia compute node:
```
module load visualization/paraview/paraview-5.13.0-RC1-EGL
pvserver --server-port=11111
```
then connect to localhost:11111 in ParaView's "Connect" dialog box
