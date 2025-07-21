#!/usr/bin/env python3
"""
Create a PVD (ParaView Data) file for time series animation
This creates a single file that ParaView can open to load all VTS files as a time series.
"""

import os
import glob
import re

def create_pvd_file(input_dir=".", output_file="lbm_simulation.pvd", dt=1.0):
    """
    Create a PVD file that references all VTS files in time order
    
    Args:
        input_dir: Directory containing VTS files
        output_file: Output PVD filename
        dt: Time step size (for calculating time values)
    """
    
    # Find all VTS files
    vts_pattern = os.path.join(input_dir, "simulation_state_t*.vts")
    vts_files = sorted(glob.glob(vts_pattern))
    
    if not vts_files:
        print(f"No VTS files found matching pattern: {vts_pattern}")
        return False
    
    print(f"Found {len(vts_files)} VTS files")
    
    # Extract time steps from filenames
    time_steps = []
    for vts_file in vts_files:
        basename = os.path.basename(vts_file)
        match = re.search(r'simulation_state_t(\d+)\.vts', basename)
        if match:
            time_step = int(match.group(1))
            time_steps.append(time_step)
        else:
            print(f"Warning: Could not extract time step from {basename}")
    
    if not time_steps:
        print("No valid time steps found")
        return False
    
    # Create PVD file
    output_path = os.path.join(input_dir, output_file)
    print(f"Creating PVD file: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        
        for i, (vts_file, time_step) in enumerate(zip(vts_files, time_steps)):
            # Calculate time value (you may need to adjust this based on your simulation)
            time_value = time_step * dt
            
            # Use relative path for the VTS file
            vts_filename = os.path.basename(vts_file)
            
            f.write(f'    <DataSet timestep="{time_value}" group="" part="0" file="{vts_filename}"/>\n')
        
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    
    print(f"PVD file created with {len(vts_files)} time steps")
    print(f"Time range: {time_steps[0] * dt} to {time_steps[-1] * dt}")
    
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create PVD file for VTS time series')
    parser.add_argument('--input-dir', '-i', default='.', 
                       help='Directory containing VTS files (default: current directory)')
    parser.add_argument('--output', '-o', default='lbm_simulation.pvd',
                       help='Output PVD filename (default: lbm_simulation.pvd)')
    parser.add_argument('--dt', type=float, default=1.0,
                       help='Time step size (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create PVD file
    success = create_pvd_file(
        input_dir=args.input_dir,
        output_file=args.output,
        dt=args.dt
    )
    
    if success:
        print(f"\nPVD file creation completed!")
        print(f"You can now open '{args.output}' in ParaView for animation")
    else:
        print("\nPVD file creation failed!")

if __name__ == "__main__":
    main()
