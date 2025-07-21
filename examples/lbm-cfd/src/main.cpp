#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <string>

#undef ASCENT_ENABLED

#ifdef ASCENT_ENABLED
#include <ascent.hpp>
#include <conduit_blueprint_mpi.hpp>
#endif

#include "lbm_mpi.hpp"
#include "paraview_sim.hpp"

void runLbmCfdSimulation(int rank, int num_ranks, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, uint32_t time_steps, void *ptr, LbmDQ::LatticeType lattice_type);
void createDivergingColorMap(uint8_t *cmap, uint32_t size);
void exportSimulationStateToVTS(LbmDQ* lbm, const char* filename, double dt, double dx, double physical_density, uint32_t time_steps);
void exportVelocityDiagnostics(int t, int rank, int num_ranks, LbmDQ* lbm, double dt, double dx, double physical_density, uint32_t time_steps);

#ifdef ASCENT_ENABLED
void updateAscentData(int rank, int num_ranks, int step, double time, conduit::Node &mesh);
void runAscentInSituTasks(conduit::Node &mesh, conduit::Node &selections, ascent::Ascent *ascent_ptr);
void repartitionCallback(conduit::Node &params, conduit::Node &output);
void steeringCallback(conduit::Node &params, conduit::Node &output);
#endif
int32_t readFile(const char *filename, char** data_ptr);

// global vars for LBM and Barriers
std::vector<Barrier*> barriers;
LbmDQ *lbm;

int main(int argc, char **argv) {
    int rc, rank, num_ranks;
    rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (rc != 0)
    {
	fprintf(stderr, "Error initializing MPI");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }


    uint32_t dim_x = 75;
    uint32_t dim_y = 75;
    uint32_t dim_z = 75;
    uint32_t time_steps = 5000;


    LbmDQ::LatticeType lattice_type;
    bool model_specified = false;

    // parse command line arguments
    for (int i = 1; i < argc; i++) {
	if (strcmp(argv[i], "--d3q15") == 0) {
	    lattice_type = LbmDQ::D3Q15;
	    model_specified = true;
	}
	else if (strcmp(argv[i], "--d3q19") == 0) {
	    lattice_type = LbmDQ::D3Q19;
	    model_specified = true;
	}
	else if (strcmp(argv[i], "--d3q27") == 0) {
	    lattice_type = LbmDQ::D3Q27;
	    model_specified = true;
	}
    }

    if (!model_specified) {
        if (rank == 0) {
	    fprintf(stderr, "Error: Lattice model must be specified. Use --d3q15, --d3q19, or --d3q27");
 	}
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "\nLBM-CFD> running with " << num_ranks << " processes" << std::endl;
        std::cout << "LBM-CFD> resolution=" << dim_x << "x" << dim_y << "x" << dim_z << ", time steps=" << time_steps << std::endl;
        std::cout << "LBM-CFD> using " << (lattice_type == LbmDQ::D3Q15 ? "D3Q15" : (lattice_type == LbmDQ::D3Q19 ? "D3Q19" : "D3Q27")) << " lattice model" << std::endl;
    }

    void *ascent_ptr = NULL;

#ifdef ASCENT_ENABLED
    if (rank == 0) std::cout << "LBM-CFD> Ascent in situ: ENABLED" << std::endl;
    
    // Copy MPI Communicator to use with Ascent
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    // Create Ascent object
    ascent::Ascent ascent;

    // Set Ascent options
    conduit::Node ascent_opts;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);

    ascent::register_callback("repartitionCallback", repartitionCallback);
    ascent::register_callback("steeringCallback", steeringCallback);

    ascent_ptr = &ascent;
#endif

    // Run simulation
    runLbmCfdSimulation(rank, num_ranks, dim_x, dim_y, dim_z, time_steps, ascent_ptr, lattice_type);

#ifdef ASCENT_ENABLED
    ascent.close();
#endif

    MPI_Finalize();

    return 0;
}

void runLbmCfdSimulation(int rank, int num_ranks, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, uint32_t time_steps, void *ptr, LbmDQ::LatticeType lattice_type)
{
    // simulate corn syrup at 25 C in a 2 m pipe, moving 0.25 m/s for 8 sec
    double physical_density = 1380.0;     // kg/m^3
    double physical_speed = 0.25;         // m/s
    double physical_length = 2.0;         // m
    double physical_viscosity = 1.3806;   // Pa s
    double kinematic_viscosity = physical_viscosity / physical_density;

    double reynolds_number = (physical_density * physical_speed * physical_length) / physical_viscosity;

    double dx = physical_length / (double)dim_x;

    // Calculate time step based on CFL condition
    double cfl_max = 0.3;  // Conservative CFL number
    double dt_cfl = cfl_max * dx / physical_speed;

    // Calculate diffusion stability limit
    double dt_diffusion = 0.5 * dx * dx / kinematic_viscosity;

    // Use the more restrictive (smaller) time step
    double dt = std::min(dt_cfl, dt_diffusion);

    // Calculate lattice parameters
    double lattice_cs = 1.0 / sqrt(3.0);  // Lattice speed of sound
    double lattice_viscosity = kinematic_viscosity * dt / (dx * dx);
    double lattice_speed = physical_speed * dt / dx;
    double mach_number = lattice_speed / lattice_cs;

    // Safety checks and corrections
    if (lattice_viscosity < 0.005) {
        double correction = 0.005 / lattice_viscosity;    
        dt *= correction;
        lattice_viscosity = kinematic_viscosity * dt / (dx * dx);
        lattice_speed = physical_speed * dt / dx;
        mach_number = lattice_speed / lattice_cs;
        if (rank == 0) {
            printf("[INFO] Reduced dt by factor %.3f for viscosity stability\n", correction);
        }
    }

    if (mach_number > 0.1) {
        double correction = 0.1 / mach_number;
        dt *= correction;
        lattice_viscosity = kinematic_viscosity * dt / (dx * dx);
        lattice_speed = physical_speed * dt / dx;
        mach_number = lattice_speed / lattice_cs;
        if (rank == 0) {
            printf("[INFO] Reduced dt by factor %.3f for Mach number stability\n", correction);
        }
    }

    // Calculate total simulated time
    double total_simulated_time = time_steps * dt;

    // Output simulation properties
    if (rank == 0)
    {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "LBM-CFD> TIME STEP CALCULATION:" << std::endl;
        std::cout << "  Grid spacing (dx): " << dx << " m" << std::endl;
        std::cout << "  CFL time step: " << dt_cfl << " s" << std::endl;
        std::cout << "  Diffusion time step: " << dt_diffusion << " s" << std::endl;
        std::cout << "  Final time step (dt): " << dt << " s" << std::endl;
        std::cout << "  Limiting factor: " << ((dt_cfl < dt_diffusion) ? "CFL" : "Diffusion") << std::endl;
        
        std::cout << "LBM-CFD> PHYSICAL PARAMETERS:" << std::endl;
        std::cout << "  density: " << physical_density << " kg/m^3" << std::endl;
        std::cout << "  viscosity: " << physical_viscosity << " Pa s" << std::endl;
        std::cout << "  speed: " << physical_speed << " m/s" << std::endl;
        std::cout << "  length: " << physical_length << " m" << std::endl;
        std::cout << "  Reynolds number: " << reynolds_number << std::endl;
        std::cout << "  kinematic viscosity: " << kinematic_viscosity << " m^2/s" << std::endl;
        
        std::cout << "LBM-CFD> LATTICE PARAMETERS:" << std::endl;
        std::cout << "  lattice speed: " << lattice_speed << std::endl;
        std::cout << "  lattice viscosity: " << lattice_viscosity << std::endl;
        std::cout << "  Mach number: " << mach_number << std::endl;
        std::cout << "  CFL number: " << cfl_max << std::endl;
        std::cout << "  Total simulated time: " << total_simulated_time << " s" << std::endl;
        
        // Stability warnings
        if (lattice_viscosity < 0.005) {
            std::cout << "*** WARNING: Low lattice viscosity! Simulation may be unstable\n";
        }
        if (mach_number > 0.1) {
            std::cout << "*** WARNING: High Mach number! Simulation may be unstable\n";
        }
        if (lattice_speed > 0.1) {
            std::cout << "*** WARNING: High lattice speed! Simulation may be unstable\n";
        }

    }

    // create LBM object
    lbm = new LbmDQ(dim_x, dim_y, dim_z, dt / dx, rank, num_ranks, lattice_type);

    // BARRIER OPTION ONE: Diamond-shaped (square rotated 45°) bluff body barriers
    barriers.clear();

    // Configuration parameters
    int diamond_x_center = std::max(10, (int)(dim_x / 6));  // Position at 1/6 of domain
    int diamond_y_center = dim_y / 2;                       // Center vertically
    int diamond_size = std::max(5, (int)(dim_y / 8));       // Size based on domain height

    // Optional: Create multiple diamonds for more complex flow
    bool use_multiple_diamonds = true;

    if (use_multiple_diamonds) {
        // Create three diamond obstacles at different positions
        std::vector<std::pair<int, int>> diamond_centers = {
            {diamond_x_center, diamond_y_center},
            {diamond_x_center + (int)(dim_x / 3), diamond_y_center - (int)(dim_y / 6)},
            {diamond_x_center + (int)(2 * dim_x / 3), diamond_y_center + (int)(dim_y / 6)}
        };

        for (const auto& center : diamond_centers) {
            int cx = center.first;
            int cy = center.second;

            // Create diamond shape (rotated square)
            for (int k = 0; k < (int)dim_z; ++k) {
                for (int j = 0; j < (int)dim_y; ++j) {
                    for (int i = 0; i < (int)dim_x; ++i) {
                        // Diamond shape condition: |x-cx| + |y-cy| <= size
                        int dx = abs(i - cx);
                        int dy = abs(j - cy);

                        if (dx + dy <= diamond_size && i >= 0 && i < (int)dim_x &&
                            j >= 0 && j < (int)dim_y && k >= 0 && k < (int)dim_z) {
                            barriers.push_back(new Barrier3D(i, i, j, j, k, k));
                        }
                    }
                }
            }
        }
    } else {
        // Single larger diamond for simpler but still interesting flow
        int large_diamond_size = std::max(8, (int)(dim_y / 5));

        // Create single diamond shape
        for (int k = 0; k < (int)dim_z; ++k) {
            for (int j = 0; j < (int)dim_y; ++j) {
                for (int i = 0; i < (int)dim_x; ++i) {
                    // Diamond shape condition: |x-cx| + |y-cy| <= size
                    int dx = abs(i - diamond_x_center);
                    int dy = abs(j - diamond_y_center);

                    if (dx + dy <= large_diamond_size && i >= 0 && i < (int)dim_x &&
                        j >= 0 && j < (int)dim_y && k >= 0 && k < (int)dim_z) {
                        barriers.push_back(new Barrier3D(i, i, j, j, k, k));
                    }
                }
            }
        }
    }

    if (rank == 0) {
        printf("[INFO] Created diamond-shaped bluff body barriers\n");
        printf("[INFO] Configuration: %s diamonds, size=%d, center=(%d,%d)\n",
               use_multiple_diamonds ? "Multiple" : "Single",
               use_multiple_diamonds ? diamond_size : diamond_size,
               diamond_x_center, diamond_y_center);
    }

    //// BARRIER OPTION TWO: Square cylinder barriers for von Kármán vortex street
    //barriers.clear();
    //
    //// Configuration for square cylinder
    //int square_x_center = std::max(10, (int)(dim_x / 5));  // Position at 1/5 of domain
    //int square_y_center = dim_y / 2;                       // Center vertically
    //int square_half_size = std::max(4, (int)(dim_y / 10)); // Half-size of square
    //
    //// Create square cylinder (spans full z-direction)
    //for (int k = 0; k < (int)dim_z; ++k) {
    //    for (int j = square_y_center - square_half_size; j <= square_y_center + square_half_size; ++j) {
    //        for (int i = square_x_center - square_half_size; i <= square_x_center + square_half_size; ++i) {
    //            if (i >= 0 && i < (int)dim_x && j >= 0 && j < (int)dim_y && k >= 0 && k < (int)dim_z) {
    //                barriers.push_back(new Barrier3D(i, i, j, j, k, k));
    //            }
    //        }
    //    }
    //}
   
    //// BARRIER OPTION THREE: Multiple cylinders for complex wake interactions
    //barriers.clear();
    //
    //// Add multiple cylindrical obstacles to create extended transient behavior
    //int cylinder_radius = 10; // Radius for cylindrical obstacles
    //
    //// First cylinder at 1/4 length
    //double cx1 = dim_x / 4.0;
    //double cy1 = dim_y / 2.0;
    //
    //// Second cylinder at 1/2 length, offset vertically
    //double cx2 = dim_x / 2.0;
    //double cy2 = dim_y / 4.0;
    //
    //// Third cylinder at 3/4 length
    //double cx3 = 3.0 * dim_x / 4.0;
    //double cy3 = 2.0 * dim_y / 4.0;
    //
    //// Create cylindrical obstacles (spanning full z-direction)
    //int cylinder1_count = 0, cylinder2_count = 0, cylinder3_count = 0;
    //for (int k = 0; k < (int)dim_z; ++k) {
    //    for (int j = 0; j < (int)dim_y; ++j) {
    //        for (int i = 0; i < (int)dim_x; ++i) {
    //            // Cylinder 1
    //            double dx1 = i - cx1;
    //            double dy1 = j - cy1;
    //            if (dx1*dx1 + dy1*dy1 <= cylinder_radius*cylinder_radius) {
    //                barriers.push_back(new Barrier3D(i, i, j, j, k, k));
    //    	    cylinder1_count++;
    //            }
    //            
    //            // Cylinder 2
    //            double dx2 = i - cx2;
    //            double dy2 = j - cy2;
    //            if (dx2*dx2 + dy2*dy2 <= cylinder_radius*cylinder_radius) {
    //                barriers.push_back(new Barrier3D(i, i, j, j, k, k));
    //    	    cylinder2_count++;
    //            }
    //            
    //            // Cylinder 3
    //            double dx3 = i - cx3;
    //            double dy3 = j - cy3;
    //            if (dx3*dx3 + dy3*dy3 <= cylinder_radius*cylinder_radius) {
    //                barriers.push_back(new Barrier3D(i, i, j, j, k, k));
    //    	    cylinder3_count++;
    //            }
    //        }
    //    }
    //}

    lbm->initBarrier(barriers);
    lbm->initFluid(physical_speed);
    lbm->checkGuards();

    // sync all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // run simulation
    int t;
    int output_count = 0;
    double output_frequency = 1.0;  // Output every 1.0 seconds
    double next_output_time = 0.0;
    uint8_t stable, all_stable = 0;

    // Timing variables
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<double> collide_time(0), stream_time(0), exchange_time(0);
    std::chrono::duration<double> total_iteration_time(0);

#ifdef ASCENT_ENABLED
    int i;
    conduit::Node selections;
#endif
    
    for (t = 0; t < time_steps; t++)
    {
        // enforce inlet boundary at every step
	lbm->updateFluid(physical_speed);
	// Output data at regular intervals
        double current_time = t * dt;  // Current simulation time
        if (current_time >= next_output_time)
	{
            if (rank == 0)
            {
	        std::cout << std::fixed << std::setprecision(3) << "LBM-CFD> time: " << current_time << " / " <<
                             total_simulated_time << " s, time step: " << t << " / " << time_steps << std::endl;
	    }
            stable = lbm->checkStability();
            MPI_Reduce(&stable, &all_stable, 1, MPI_UNSIGNED_CHAR, MPI_MAX, 0, MPI_COMM_WORLD);
            if (!all_stable && rank == 0)
            {
                fprintf(stderr, "LBM-CFD> Warning: simulation has become unstable (more time steps needed)");
            }
           
#ifdef ASCENT_ENABLED
            ascent::Ascent *ascent_ptr = static_cast<ascent::Ascent*>(ptr);
            conduit::Node mesh;
            updateAscentData(rank, num_ranks, t, time, mesh);
            runAscentInSituTasks(mesh, selections, ascent_ptr);
#endif
            output_count++;
            next_output_time = output_count * output_frequency;
        }

	// Export vorticity for visualization
#if OUTPUT_VORTICITY
        printSimulationDiagnostics(t, rank, lbm, dt, dx, physical_density, time_steps);
#endif
        
        // Export velocity vectors for streamline visualization
#if OUTPUT_VELOCITY
        exportVelocityDiagnostics(t, rank, num_ranks, lbm, dt, dx, physical_density, time_steps);
#endif

	// Time the entire iteration
        start_time = std::chrono::high_resolution_clock::now();

        // Time collide step
        auto collide_start = std::chrono::high_resolution_clock::now();

	lbm->collide(lattice_viscosity, t);

	auto collide_end = std::chrono::high_resolution_clock::now();
        collide_time += std::chrono::duration_cast<std::chrono::duration<double>>(collide_end - collide_start);
       
	// Time stream step
        auto stream_start = std::chrono::high_resolution_clock::now();
	
	lbm->stream();

	auto stream_end = std::chrono::high_resolution_clock::now();
        stream_time += std::chrono::duration_cast<std::chrono::duration<double>>(stream_end - stream_start);

	// Time exchangeBoundaries step
        auto exchange_start = std::chrono::high_resolution_clock::now();

        lbm->exchangeBoundaries();

	auto exchange_end = std::chrono::high_resolution_clock::now();
        exchange_time += std::chrono::duration_cast<std::chrono::duration<double>>(exchange_end - exchange_start);

        end_time = std::chrono::high_resolution_clock::now();
        total_iteration_time += std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    }

    // Print final timing summary
    if (rank == 0)
    {
        std::cout << "\n=== FINAL TIMING SUMMARY ===" << std::endl;
        std::cout << std::left << std::setw(15) << "Metric" << "\t" 
                  << std::right << std::setw(12) << "Value" << "\t" 
                  << std::left << std::setw(10) << "Unit" << "\t" 
                  << std::right << std::setw(10) << "Percentage" << std::endl;
        std::cout << std::left << std::setw(15) << "Total steps" << "\t" 
                  << std::right << std::setw(12) << time_steps << "\t" 
                  << std::left << std::setw(10) << "steps" << "\t" 
                  << std::right << std::setw(10) << "" << std::endl;
        std::cout << std::left << std::setw(15) << "Collide" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(6) << collide_time.count() << "\t" 
                  << std::left << std::setw(10) << "seconds" << "\t" 
                  << std::right << std::setw(10) << std::setprecision(2) << (collide_time.count() / total_iteration_time.count()) * 100 << "%" << std::endl;
        std::cout << std::left << std::setw(15) << "Stream" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(6) << stream_time.count() << "\t" 
                  << std::left << std::setw(10) << "seconds" << "\t" 
                  << std::right << std::setw(10) << std::setprecision(2) << (stream_time.count() / total_iteration_time.count()) * 100 << "%" << std::endl;
        std::cout << std::left << std::setw(15) << "Exchange" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(6) << exchange_time.count() << "\t" 
                  << std::left << std::setw(10) << "seconds" << "\t" 
                  << std::right << std::setw(10) << std::setprecision(2) << (exchange_time.count() / total_iteration_time.count()) * 100 << "%" << std::endl;
        std::cout << std::left << std::setw(15) << "Total" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(6) << total_iteration_time.count() << "\t" 
                  << std::left << std::setw(10) << "seconds" << "\t" 
                  << std::right << std::setw(10) << "" << std::endl;
        std::cout << std::left << std::setw(15) << "Avg per step" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(6) << total_iteration_time.count() / time_steps << "\t" 
                  << std::left << std::setw(10) << "seconds" << "\t" 
                  << std::right << std::setw(10) << "" << std::endl;
        std::cout << std::left << std::setw(15) << "Steps/sec" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2) << time_steps / total_iteration_time.count() << "\t" 
                  << std::left << std::setw(10) << "steps/sec" << "\t" 
                  << std::right << std::setw(10) << "" << std::endl;
        std::cout << "========================" << std::endl;
    }

    // Clean up    
    delete lbm;
    // Delete all Barrier* objects to prevent memory leaks
    for (Barrier* b : barriers) {
        delete b;
    }
    barriers.clear();
}

#ifdef ASCENT_ENABLED
void updateAscentData(int rank, int num_ranks, int step, double time, conduit::Node &mesh)
{
    // Gather data on rank 0
    lbm->computeVorticity();

    uint32_t dim_x = lbm->getDimX();
    uint32_t dim_y = lbm->getDimY();
    uint32_t offset_x = lbm->getOffsetX();
    uint32_t offset_y = lbm->getOffsetY();
    uint32_t prop_size = dim_x * dim_y;

    int *barrier_data = new int[barriers.size() * 4];
    int i;
    for (i = 0; i < barriers.size(); i++)
    {
        barrier_data[4 * i + 0] = barriers[i]->getX1();
        barrier_data[4 * i + 1] = barriers[i]->getY1();
        barrier_data[4 * i + 2] = barriers[i]->getX2();
        barrier_data[4 * i + 3] = barriers[i]->getY2();
    }

    uint32_t start_x = lbm->getStartX();
    uint32_t start_y = lbm->getStartY();

    mesh["state/domain_id"] = rank;
    mesh["state/num_domains"] = num_ranks;
    mesh["state/cycle"] = step;
    mesh["state/time"] = time;
    mesh["state/coords/start/x"] = lbm->getStartX();
    mesh["state/coords/start/y"] = lbm->getStartY();
    mesh["state/coords/size/x"] = lbm->getSizeX();
    mesh["state/coords/size/y"] = lbm->getSizeY();
    mesh["state/num_barriers"] = barriers.size();
    mesh["state/barriers"].set(barrier_data, barriers.size() * 4);
    
    mesh["coordsets/coords/type"] = "uniform";
    mesh["coordsets/coords/dims/i"] = dim_x + 1;
    mesh["coordsets/coords/dims/j"] = dim_y + 1;

    mesh["coordsets/coords/origin/x"] = offset_x - start_x;
    mesh["coordsets/coords/origin/y"] = offset_y - start_y;
    mesh["coordsets/coords/spacing/dx"] = 1;
    mesh["coordsets/coords/spacing/dy"] = 1;

    mesh["topologies/topo/type"] = "uniform";
    mesh["topologies/topo/coordset"] = "coords";
    
    mesh["fields/vorticity/association"] = "element";
    mesh["fields/vorticity/topology"] = "topo";
    mesh["fields/vorticity/values"].set_external(lbm->getVorticity(), prop_size);


    conduit::Node options, selections, output;
    for (i = 0; i < num_ranks; i++)
    {
        uint32_t *rank_start = lbm->getRankLocalStart(i);
        uint32_t *rank_size = lbm->getRankLocalSize(i);
        conduit::Node &selection = selections.append();
        selection["type"] = "logical";
        selection["domain_id"] = i;
        selection["start"] = {rank_start[0], rank_start[1], 0u};
        selection["end"] = {rank_start[0] + rank_size[0] - 1u, rank_start[1] + rank_size[1] - 1u, 0u};
    }
    options["target"] = 1;
    options["fields"] = {"vorticity"};
    options["selections"] = selections;
    options["mapping"] = 0;

    conduit::blueprint::mpi::mesh::partition(mesh, options, output, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        mesh["coordsets/coords_whole/"] = output["coordsets/coords"];

        mesh["topologies/topo_whole/type"] = "uniform";
        mesh["topologies/topo_whole/coordset"] = "coords_whole";

        mesh["fields/vorticity_whole/association"] = "element";
        mesh["fields/vorticity_whole/topology"] = "topo_whole";
        mesh["fields/vorticity_whole/values"] = output["fields/vorticity/values"];
    }


    delete[] barrier_data;
}

void runAscentInSituTasks(conduit::Node &mesh, conduit::Node &selections, ascent::Ascent *ascent_ptr)
{
    ascent_ptr->publish(mesh);

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    char *py_script;
    if (readFile("ascent/ascent_trame_bridge.py", &py_script) >= 0)
    {
        extracts["e1/type"] = "python";
        extracts["e1/params/source"] = py_script;
    }
    //std::cout << actions.to_yaml() << std::endl;

    ascent_ptr->execute(actions);
}

void repartitionCallback(conduit::Node &params, conduit::Node &output)
{
    int num_ranks = (int)params["state/num_domains"].as_int32();
    uint32_t layout[4] = {params["state/coords/start/x"].as_uint32(), params["state/coords/start/y"].as_uint32(),
                          params["state/coords/size/x"].as_uint32(), params["state/coords/size/y"].as_uint32()};
    uint32_t *layout_all = new uint32_t[4 * num_ranks];
    MPI_Allgather(layout, 4, MPI_UNSIGNED, layout_all, 4, MPI_UNSIGNED, MPI_COMM_WORLD);

    int i;
    conduit::Node options, selections;
    for (i = 0; i < num_ranks; i++)
    {
        uint32_t rank_start_x = layout_all[4 * i];
        uint32_t rank_start_y = layout_all[4 * i + 1];
        uint32_t rank_size_x = layout_all[4 * i + 2];
        uint32_t rank_size_y = layout_all[4 * i + 3];
        conduit::Node &selection = selections.append();
        selection["type"] = "logical";
        selection["domain_id"] = i;
        selection["start"] = {rank_start_x, rank_start_y, 0u};
        selection["end"] = {rank_start_x + rank_size_x - 1u, rank_start_y + rank_size_y - 1u, 0u};
    }
    options["target"] = 1;
    options["fields"] = {"vorticity"};
    options["selections"] = selections;
    options["mapping"] = 0;

    conduit::blueprint::mpi::mesh::partition(params, options, output, MPI_COMM_WORLD);

    delete[] layout_all;
}

void steeringCallback(conduit::Node &params, conduit::Node &output)
{
    if (params.has_path("task_id") && params.has_path("flow_speed") && params.has_path("num_barriers") && params.has_path("barriers"))
    {
        int rank = (int)params["task_id"].as_int64();
        double flow_speed = params["flow_speed"].as_float64();
        int num_barriers = (int)params["num_barriers"].as_int64();
        int32_t *new_barriers = params["barriers"].as_int32_ptr();
        
        int i;
        barriers.clear();
        for (i = 0; i < num_barriers; i++)
        {
            int x1 = new_barriers[4 * i + 0];
            int y1 = new_barriers[4 * i + 1];
            int x2 = new_barriers[4 * i + 2];
            int y2 = new_barriers[4 * i + 3];
            if (x1 == x2)
            {
                barriers.push_back(new Barrier3D(x1, x2, y1, y2, 0, 0));
	    }
            else if (y1 == y2)
            {
                barriers.push_back(new Barrier3D(x1, x2, y1, y2, 0, 0));
            }
        }
        lbm->initBarrier(barriers);
        lbm->updateFluid(flow_speed);
    }
}
#endif

int32_t readFile(const char *filename, char** data_ptr)
{
    FILE *fp;
    int err = 0;
#ifdef _WIN32
    err = fopen_s(&fp, filename, "rb");
#else
    fp = fopen(filename, "rb");
#endif
    if (err != 0 || fp == NULL)
    {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    int32_t fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (fsize <= 0) {
        std::cerr << "Error: file size is zero or negative for " << filename << std::endl;
        fclose(fp);
        return -1;
    }

    *data_ptr = (char*)malloc(fsize + 1);
    size_t read = fread(*data_ptr, fsize, 1, fp);
    if (read != 1)
    {
        std::cerr << "Error: cannot read " << filename <<std::endl;
        return -1;
    }
    (*data_ptr)[fsize] = '\0';

    fclose(fp);

    return fsize;
}
