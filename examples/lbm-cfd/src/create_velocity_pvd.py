#!/usr/bin/env python3
"""
Create PVD file for velocity vector VTS files to enable time-series visualization in ParaView.
This script creates a PVD file that references all velocity_vectors_t*.vts files for streamline visualization.
"""

import os
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom
import zipfile
import shutil
import argparse

def create_velocity_pvd_file(vts_directory=".", output_file="velocity_vectors.pvd", compress=False, archive_name="velocity_vectors.zip"):
    """
    Create a PVD file that references all velocity vector VTS files.
    
    Args:
        vts_directory: Directory containing the velocity_vectors_t*.vts files
        output_file: Output PVD filename
        compress: Whether to compress VTS files into a zip archive
        archive_name: Name of the zip archive if compression is enabled
    """
    
    # Find all velocity vector VTS files
    vts_pattern = os.path.join(vts_directory, "velocity_vectors_t*.vts")
    vts_files = glob.glob(vts_pattern)
    
    if not vts_files:
        print(f"No velocity vector VTS files found in {vts_directory}")
        print(f"Looking for pattern: {vts_pattern}")
        return
    
    # Sort files by timestep
    vts_files.sort()
    
    print(f"Found {len(vts_files)} velocity vector VTS files")
    
    # Calculate total file sizes
    total_size = sum(os.path.getsize(f) for f in vts_files)
    print(f"Total size of VTS files: {total_size / (1024*1024):.1f} MB")
    
    # Create compression archive if requested
    if compress:
        create_compressed_archive(vts_files, archive_name)
    
    # Create PVD XML structure
    root = ET.Element("VTKFile")
    root.set("type", "Collection")
    root.set("version", "0.1")
    root.set("byte_order", "LittleEndian")
    
    collection = ET.SubElement(root, "Collection")
    
    # Add each VTS file to the collection
    for vts_file in vts_files:
        # Extract timestep from filename (e.g., velocity_vectors_t01000.vts -> 1000)
        basename = os.path.basename(vts_file)
        try:
            # Extract timestep number from filename
            timestep_str = basename.split('_t')[1].split('.')[0]
            timestep = int(timestep_str)
        except (IndexError, ValueError):
            print(f"Warning: Could not extract timestep from {basename}, skipping")
            continue
        
        # Create DataSet element
        dataset = ET.SubElement(collection, "DataSet")
        dataset.set("timestep", str(timestep))
        dataset.set("file", os.path.basename(vts_file))
        
        print(f"Added {basename} with timestep {timestep}")
    
    # Write PVD file with pretty formatting
    rough_string = ET.tostring(root, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Remove empty lines
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(pretty_xml)
    
    print(f"Created PVD file: {output_file}")
    
    if compress:
        print(f"Created compressed archive: {archive_name}")
        archive_size = os.path.getsize(archive_name)
        compression_ratio = (1 - archive_size / total_size) * 100
        print(f"Archive size: {archive_size / (1024*1024):.1f} MB")
        print(f"Compression ratio: {compression_ratio:.1f}%")
        print(f"You can download the single archive file: {archive_name}")
    
    print(f"You can now open {output_file} in ParaView for streamline visualization")
    
    # Print instructions
    print("\nInstructions for ParaView streamline visualization:")
    print("1. Open the PVD file in ParaView")
    print("2. Apply 'Stream Tracer' filter")
    print("3. Set 'Vectors' to 'velocity'")
    print("4. Configure seed points (line, plane, or point source)")
    print("5. Play the animation to see particle paths over time")
    
    if compress:
        print("\nFile Management:")
        print("- The PVD file references individual VTS files, so keep them uncompressed for ParaView")
        print("- The compressed archive is for download/storage purposes")
        print("- You can safely delete individual VTS files after creating the archive if needed")

def create_compressed_archive(vts_files, archive_name):
    """
    Create a compressed ZIP archive containing all VTS files.
    
    Args:
        vts_files: List of VTS file paths
        archive_name: Name of the output ZIP file
    """
    print(f"Creating compressed archive: {archive_name}")
    
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for vts_file in vts_files:
            # Add file to zip with just the filename (no path)
            basename = os.path.basename(vts_file)
            print(f"  Adding {basename} to archive...")
            zipf.write(vts_file, basename)
    
    print(f"Archive creation completed: {archive_name}")

def create_simulation_state_archive(directory=".", archive_name="simulation_data.zip"):
    """
    Create a comprehensive archive containing both velocity vectors and simulation state files.
    
    Args:
        directory: Directory containing VTS files
        archive_name: Name of the output ZIP file
    """
    print("Creating comprehensive simulation data archive...")
    
    # Find all VTS files (both velocity and simulation state)
    velocity_pattern = os.path.join(directory, "velocity_vectors_t*.vts")
    state_pattern = os.path.join(directory, "simulation_state_t*.vts")
    
    velocity_files = glob.glob(velocity_pattern)
    state_files = glob.glob(state_pattern)
    
    all_files = velocity_files + state_files
    
    if not all_files:
        print("No VTS files found for archiving")
        return
    
    # Calculate total size
    total_size = sum(os.path.getsize(f) for f in all_files)
    print(f"Total files: {len(all_files)} ({len(velocity_files)} velocity, {len(state_files)} state)")
    print(f"Total size: {total_size / (1024*1024):.1f} MB")
    
    # Create archive
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for vts_file in all_files:
            basename = os.path.basename(vts_file)
            print(f"  Adding {basename}...")
            zipf.write(vts_file, basename)
    
    # Report compression results
    archive_size = os.path.getsize(archive_name)
    compression_ratio = (1 - archive_size / total_size) * 100
    print(f"Archive created: {archive_name}")
    print(f"Archive size: {archive_size / (1024*1024):.1f} MB")
    print(f"Compression ratio: {compression_ratio:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create PVD file for velocity vector VTS files")
    parser.add_argument("--directory", "-d", default=".", help="Directory containing VTS files")
    parser.add_argument("--output", "-o", default="velocity_vectors.pvd", help="Output PVD filename")
    parser.add_argument("--compress", "-c", action="store_true", help="Create compressed ZIP archive")
    parser.add_argument("--archive", "-a", default="velocity_vectors.zip", help="Archive filename")
    parser.add_argument("--all", action="store_true", help="Archive all VTS files (velocity + simulation state)")
    parser.add_argument("--all-archive", default="simulation_data.zip", help="Archive filename for all files")
    
    args = parser.parse_args()
    
    if args.all:
        # Create comprehensive archive of all simulation data
        create_simulation_state_archive(args.directory, args.all_archive)
    else:
        # Create PVD file for velocity vectors (with optional compression)
        create_velocity_pvd_file(args.directory, args.output, args.compress, args.archive)
