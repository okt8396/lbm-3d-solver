#!/usr/bin/env python3
"""
LBM-CFD PVD Creator from VTS Files
Creates ParaView Data (PVD) files for both vorticity and velocity visualization.
Auto-detects available VTS files and creates appropriate PVD collections.
"""

import os
import re
import argparse
import sys
from pathlib import Path

def extract_timestep(filename):
    """Extract timestep from VTS filename."""
    match = re.search(r't(\d+)\.vts$', filename)
    return int(match.group(1)) if match else None

def create_pvd_file(vts_files, output_file, data_type):
    """Create a PVD file from a list of VTS files."""
    if not vts_files:
        return False
    
    # Sort files by timestep
    vts_files.sort(key=lambda x: extract_timestep(x[0]) or 0)
    
    print(f"Creating {data_type} PVD file: {output_file}")
    print(f"Found {len(vts_files)} {data_type} VTS files")
    
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        
        for filename, timestep in vts_files:
            f.write(f'    <DataSet timestep="{float(timestep)}" group="" part="0" file="{filename}"/>\n')
        
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    
    timesteps = [timestep for _, timestep in vts_files]
    print(f"Time range: {min(timesteps)} to {max(timesteps)}")
    print(f"PVD file created successfully: {output_file}")
    return True

def scan_vts_files(input_dir):
    """Scan directory for VTS files and categorize them."""
    vorticity_files = []
    velocity_files = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.vts'):
            timestep = extract_timestep(filename)
            if timestep is not None:
                if filename.startswith('simulation_state_'):
                    vorticity_files.append((filename, timestep))
                elif filename.startswith('velocity_vectors_'):
                    velocity_files.append((filename, timestep))
    
    return vorticity_files, velocity_files

def print_paraview_instructions():
    """Print detailed ParaView usage instructions."""
    print("=" * 60)
    print("PARAVIEW VISUALIZATION INSTRUCTIONS")
    print("=" * 60)
    print()
    print("VORTICITY VISUALIZATION (lbm_vorticity.pvd):")
    print("1. Open ParaView")
    print("2. File → Open → Select 'lbm_vorticity.pvd'")
    print("3. Click 'Apply' in the Properties panel")
    print("4. For better visualization:")
    print("   - Change coloring to 'vorticity'")
    print("   - Adjust color scale range")
    print("   - Try 'Contour' filter for iso-surfaces")
    print("   - Use 'Slice' filter for cross-sections")
    print("5. Use the play button to animate through time")
    print()
    print("VELOCITY VISUALIZATION (velocity_vectors.pvd):")
    print("1. Open ParaView")
    print("2. File → Open → Select 'velocity_vectors.pvd'")
    print("3. Click 'Apply' in the Properties panel")
    print("4. For streamline visualization:")
    print("   - Apply 'Stream Tracer' filter")
    print("   - Set 'Vectors' to 'velocity'")
    print("   - Configure seed points (line, plane, or point source)")
    print("   - Adjust 'Maximum Streamline Length'")
    print("5. Use the play button to animate through time")
    print()
    print("Tip: You can load multiple PVD files simultaneously to compare results!")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Create PVD files from VTS files for LBM-CFD visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_pvd_from_vts.py --input-dir ./paraview
  python create_pvd_from_vts.py --input-dir . --vorticity-output custom_vorticity.pvd
        """
    )
    
    parser.add_argument('--input-dir', '-i', 
                       default='.', 
                       help='Directory containing VTS files (default: current directory)')
    
    parser.add_argument('--vorticity-output', 
                       default='lbm_vorticity.pvd',
                       help='Output filename for vorticity PVD (default: lbm_vorticity.pvd)')
    
    parser.add_argument('--velocity-output', 
                       default='velocity_vectors.pvd',
                       help='Output filename for velocity PVD (default: velocity_vectors.pvd)')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    input_dir = os.path.abspath(args.input_dir)
    
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist")
        sys.exit(1)
    
    print("LBM-CFD PVD Creator from VTS Files")
    print("=" * 40)
    print(f"Scanning directory: {input_dir}")
    
    # Scan for VTS files
    vorticity_files, velocity_files = scan_vts_files(input_dir)
    
    created_files = 0
    
    # Create vorticity PVD if files exist
    if vorticity_files:
        vorticity_path = os.path.join(input_dir, args.vorticity_output)
        if create_pvd_file(vorticity_files, vorticity_path, "vorticity"):
            created_files += 1
    
    # Create velocity PVD if files exist  
    if velocity_files:
        velocity_path = os.path.join(input_dir, args.velocity_output)
        if create_pvd_file(velocity_files, velocity_path, "velocity"):
            created_files += 1
    
    # Summary
    if created_files > 0:
        print(f"Successfully created {created_files} PVD file(s):")
        if vorticity_files:
            print(f"   - {args.vorticity_output} (vorticity)")
        if velocity_files:
            print(f"   - {args.velocity_output} (velocity)")
        print_paraview_instructions()
    else:
        print("No VTS files found. No PVD files created.")
        print("Make sure you have VTS files with names like:")
        print("  - simulation_state_t00000.vts (for vorticity)")
        print("  - velocity_vectors_t00000.vts (for velocity)")

if __name__ == "__main__":
    main()
