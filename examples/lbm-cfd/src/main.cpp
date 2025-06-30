#include <chrono>
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
#include "spdlog/spdlog.h"

void runLbmCfdSimulation(int rank, int num_ranks, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, uint32_t time_steps, void *ptr, LbmDQ::LatticeType lattice_type);
void createDivergingColorMap(uint8_t *cmap, uint32_t size);
void exportSimulationStateToFile(LbmDQ* lbm, const char* filename);
void exportSimulationStateToVTK(LbmDQ* lbm, const char* filename);

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
    spdlog::set_level(spdlog::level::warn);
    int rc, rank, num_ranks;
    rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (rc != 0)
    {
	spdlog::error("Error initializing MPI");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    uint32_t dim_x = 60;
    uint32_t dim_y = 60;
    uint32_t dim_z = 60;
    uint32_t time_steps = 20000;
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
	    spdlog::error("Error: Lattice model must be specified. Use --d3q15, --d3q19, or --d3q27");
 	}
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "LBM-CFD> running with " << num_ranks << " processes" << std::endl;
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
//    std::cout << "[Rank " << rank << "] Returned from runLbmCfdSimulation." << std::endl;

#ifdef ASCENT_ENABLED
    ascent.close();
#endif

//    std::cout << "[Rank " << rank << "] About to call MPI_Barrier before MPI_Finalize." << std::endl;
//    MPI_Barrier(MPI_COMM_WORLD);
//    std::cout << "[Rank " << rank << "] Passed MPI_Barrier." << std::endl;

//    std::cout << "[Rank " << rank << "] About to call MPI_Finalize." << std::endl;
    MPI_Finalize();
//    std::cout << "[Rank " << rank << "] MPI_Finalize completed." << std::endl;

    return 0;
}

void runLbmCfdSimulation(int rank, int num_ranks, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, uint32_t time_steps, void *ptr, LbmDQ::LatticeType lattice_type)
{
    // simulate corn syrup at 25 C in a 2 m pipe, moving 0.25 m/s for 8 sec
    double physical_density = 1380.0;     // kg/m^3
    double physical_speed = 0.25;         // m/s
    double physical_length = 2.0;         // m
    double physical_viscosity = 1.3806;   // Pa s
    double physical_time = 8.0;           // s
    double physical_freq = 0.25;//0.04;          // s
    double reynolds_number = (physical_density * physical_speed * physical_length) / physical_viscosity;

    // convert physical properties into simulation properties
    double dt = physical_time / time_steps;
    double dx = physical_length / (double)dim_y;
    double lattice_viscosity = physical_viscosity * dt / (dx * dx);
    double lattice_speed = physical_speed * dt / dx;
    double simulated_time = time_steps * dt;

    // output simulation properties
    if (rank == 0)
    {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "LBM-CFD> PHYSICAL PARAMETERS:" << std::endl;
        std::cout << "  density:   " << physical_density << " kg/m^3" << std::endl;
        std::cout << "  viscosity: " << physical_viscosity << " m^2/s" << std::endl;
        std::cout << "  speed:     " << physical_speed << " m/s" << std::endl;
        std::cout << "  length:    " << physical_length << " m" << std::endl;
        std::cout << "  time:      " << physical_time << " s" << std::endl;
        std::cout << "  Reynolds number:   " << reynolds_number << std::endl;
        if (lattice_viscosity < 0.005) {
            std::cout << "*** WARNING: lattice_viscosity < 0.005! Simulation may be unstable. Increase dt, decrease grid resolution, or increase physical viscosity.\n";
        }
	if (lattice_speed > 0.1) {
            std::cout << "*** WARNING: lattice_speed > 0.1! Simulation may be unstable. Decrease physical_speed or increase grid resolution.\n";
	}
    }
    
    // create LBM object
    lbm = new LbmDQ(dim_x, dim_y, dim_z, dt / dx, rank, num_ranks, lattice_type);

    // initialize simulation
    int zmin = 0, zmax = dim_z - 1;

    // OLD barriers (assymetric)
    //barriers.push_back(new Barrier3D(dim_x / 8, dim_x / 8, 8 * dim_y / 27 + 1, 12 * dim_y / 27 - 1, zmin, zmax));
    //barriers.push_back(new Barrier3D(dim_x / 8 + 1, dim_x / 8 + 1, 8 * dim_y / 27 + 1, 12 * dim_y / 27 - 1, zmin, zmax));
    //barriers.push_back(new Barrier3D(dim_x / 8, dim_x / 8, 13 * dim_y / 27 + 1, 17 * dim_y / 27 - 1, zmin, zmax));
    //barriers.push_back(new Barrier3D(dim_x / 8 + 1, dim_x / 8 + 1, 13 * dim_y / 27 + 1, 17 * dim_y / 27 - 1, zmin, zmax));
    
    // NEW barriers (central sphere)
    barriers.clear();
    // Add a central sphere barrier
    double cx = (dim_x - 1) / 2.0;
    double cy = (dim_y - 1) / 2.0;
    double cz = (dim_z - 1) / 2.0;
    int radius = std::min({dim_x, dim_y, dim_z}) / 5; // 1/5 of the smallest dimension
    for (int k = 0; k < (int)dim_z; ++k) {
        for (int j = 0; j < (int)dim_y; ++j) {
            for (int i = 0; i < (int)dim_x; ++i) {
                double dx = i - cx;
                double dy = j - cy;
                double dz = k - cz;
                if (dx*dx + dy*dy + dz*dz <= radius*radius) {
                    barriers.push_back(new Barrier3D(i, i, j, j, k, k));
                }
            }
        }
    }
    lbm->initBarrier(barriers);
    lbm->initFluid(physical_speed);

//    std::cout << "[Rank " << rank << "] Finished lbm->initFluid, about to checkGuards()" << std::endl;
    lbm->checkGuards();
//    std::cout << "[Rank " << rank << "] Finished checkGuards(), about to call MPI_Barrier" << std::endl;

    // sync all processes
    MPI_Barrier(MPI_COMM_WORLD);
//    std::cout << "[Rank " << rank << "] Finished MPI_Barrier, about to enter simulation loop" << std::endl;

    // --- BEGIN: Barrier-adjacent probe debug block ---
    const int N = 3; // Max nodes to track per rank
    const int MAX_PRINT_RANK = 1; // Only ranks < MAX_PRINT_RANK will print (set to 1 for only rank 0)
    struct NodeProbe {
        int local_i, local_j, local_k;
        int global_X, global_Y, global_Z;
        std::vector<int> steps;
        std::vector<float> vx, vy, vz, density;
    };
    std::vector<NodeProbe> near_barrier_nodes;
    // --- END: Barrier-adjacent probe debug block ---

    // --- BEGIN: Barrier-adjacent probe search ---
    // Only allow a few ranks to track nodes
    if (rank < MAX_PRINT_RANK) {
        for (int k = 1; k < lbm->getDimZ() - 1 && (int)near_barrier_nodes.size() < N; ++k)
        for (int j = 1; j < lbm->getDimY() - 1 && (int)near_barrier_nodes.size() < N; ++j)
        for (int i = 1; i < lbm->getDimX() - 1 && (int)near_barrier_nodes.size() < N; ++i) {
            int idx = lbm->getDimX() * (j + lbm->getDimY() * k) + i;
            uint8_t* barrier = lbm->getBarrier();
            if (!barrier) barrier = lbm->getLocalBarrier();
            if (barrier[idx]) continue;
            // Check if any neighbor is a barrier
            bool near_barrier = false;
            const int (*c)[3] = lbm->getC();
            for (int d = 1; d < lbm->getQ(); ++d) {
                int ni = i + c[d][0];
                int nj = j + c[d][1];
                int nk = k + c[d][2];
                if (ni < 0 || ni >= lbm->getDimX() ||
                    nj < 0 || nj >= lbm->getDimY() ||
                    nk < 0 || nk >= lbm->getDimZ()) continue;
                int nidx = lbm->getDimX() * (nj + lbm->getDimY() * nk) + ni;
                if (barrier[nidx]) {
                    near_barrier = true;
                    break;
                }
            }
            if (near_barrier) {
                NodeProbe probe;
                probe.local_i = i;
                probe.local_j = j;
                probe.local_k = k;
                probe.global_X = i + lbm->getOffsetX();
                probe.global_Y = j + lbm->getOffsetY();
                probe.global_Z = k + lbm->getOffsetZ();
                near_barrier_nodes.push_back(probe);
            }
	}
    }
    // --- END: Barrier-adjacent probe search ---

    // run simulation
    int t;
    double time;
    int output_count = 0;
    double next_output_time = 0.0;
    uint8_t stable, all_stable = 0;

    // Timing variables
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<double> collide_time(0), stream_time(0), bounceback_time(0), exchange_time(0);
    std::chrono::duration<double> total_iteration_time(0);

#ifdef ASCENT_ENABLED
    int i;
    conduit::Node selections;
#endif
//    std::cout << "[Rank " << rank << "] About to start simulation loop (for t)" << std::endl;
    for (t = 0; t < time_steps; t++)
    {
        // output data at frequency equivalent to `physical_freq` time
        time = t * dt;
        if (time >= next_output_time)
        {
            if (rank == 0)
            {
                std::cout << std::fixed << std::setprecision(3) << "LBM-CFD> time: " << time << " / " <<
                             physical_time << " , time step: " << t << " / " << time_steps << std::endl;
            }
            stable = lbm->checkStability();
            MPI_Reduce(&stable, &all_stable, 1, MPI_UNSIGNED_CHAR, MPI_MAX, 0, MPI_COMM_WORLD);
            if (!all_stable && rank == 0)
            {
                spdlog::warn("LBM-CFD> Warning: simulation has become unstable (more time steps needed)");
            }
            
#ifdef ASCENT_ENABLED
            ascent::Ascent *ascent_ptr = static_cast<ascent::Ascent*>(ptr);
            conduit::Node mesh;
            updateAscentData(rank, num_ranks, t, time, mesh);
            runAscentInSituTasks(mesh, selections, ascent_ptr);
#endif
            output_count++;
            next_output_time = output_count * physical_freq;
        }
    
	// Time the entire iteration
        start_time = std::chrono::high_resolution_clock::now();

        // Time collide step
        auto collide_start = std::chrono::high_resolution_clock::now();

	lbm->collide(lattice_viscosity, t);
//	lbm->checkGuards();

	auto collide_end = std::chrono::high_resolution_clock::now();
        collide_time += std::chrono::duration_cast<std::chrono::duration<double>>(collide_end - collide_start);
       
	// Time stream step
        auto stream_start = std::chrono::high_resolution_clock::now();

	lbm->stream();
//	lbm->checkGuards();

	auto stream_end = std::chrono::high_resolution_clock::now();
        stream_time += std::chrono::duration_cast<std::chrono::duration<double>>(stream_end - stream_start);

	// Time bounceBackStream step
        auto bounceback_start = std::chrono::high_resolution_clock::now();

        lbm->bounceBackStream();
//	lbm->checkGuards();
   
	auto bounceback_end = std::chrono::high_resolution_clock::now();
        bounceback_time += std::chrono::duration_cast<std::chrono::duration<double>>(bounceback_end - bounceback_start);

	// Time exchangeBoundaries step
        auto exchange_start = std::chrono::high_resolution_clock::now();

        lbm->exchangeBoundaries();
//	lbm->checkGuards();

	auto exchange_end = std::chrono::high_resolution_clock::now();
        exchange_time += std::chrono::duration_cast<std::chrono::duration<double>>(exchange_end - exchange_start);

        end_time = std::chrono::high_resolution_clock::now();
        total_iteration_time += std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
	
	// After all steps for this iteration, record near-barrier node properties if needed
        if (rank < MAX_PRINT_RANK && t % 5000 == 0) {
            float* vx = lbm->getVelocityX();
            float* vy = lbm->getVelocityY();
            float* vz = lbm->getVelocityZ();
            float* density = lbm->getDensity();
            for (auto& probe : near_barrier_nodes) {
                int idx = lbm->getDimX() * (probe.local_j + lbm->getDimY() * probe.local_k) + probe.local_i;
                probe.steps.push_back(t);
                probe.vx.push_back(vx[idx]);
                probe.vy.push_back(vy[idx]);
                probe.vz.push_back(vz[idx]);
                probe.density.push_back(density[idx]);
            }
        }
    }
//    std::cout << "[Rank " << rank << "] Exited simulation loop at t=" << t << " / " << time_steps << std::endl;

    // Print final timing summary
    if (rank == 0)
    {
        std::cout << "\n=== FINAL TIMING SUMMARY ===" << std::endl;
        std::cout << "Total steps: " << time_steps << std::endl;
        std::cout << "Collide:        " << std::fixed << std::setprecision(6) << collide_time.count() << "s ("
                  << std::setprecision(2) << (collide_time.count() / total_iteration_time.count()) * 100 << "%)" << std::endl;
        std::cout << "Stream:         " << std::fixed << std::setprecision(6) << stream_time.count() << "s ("
                  << std::setprecision(2) << (stream_time.count() / total_iteration_time.count()) * 100 << "%)" << std::endl;
        std::cout << "BounceBack:     " << std::fixed << std::setprecision(6) << bounceback_time.count() << "s ("
                  << std::setprecision(2) << (bounceback_time.count() / total_iteration_time.count()) * 100 << "%)" << std::endl;
        std::cout << "Exchange:       " << std::fixed << std::setprecision(6) << exchange_time.count() << "s ("
                  << std::setprecision(2) << (exchange_time.count() / total_iteration_time.count()) * 100 << "%)" << std::endl;                                                                                                                         std::cout << "Total:          " << std::fixed << std::setprecision(6) << total_iteration_time.count() << "s" << std::endl;
        std::cout << "Avg per step:   " << std::fixed << std::setprecision(6) << total_iteration_time.count() / time_steps << "s" << std::endl;
        std::cout << "Steps/sec:      " << std::fixed << std::setprecision(2) << time_steps / total_iteration_time.count() << std::endl;
        std::cout << "========================" << std::endl;
    }

// === Gather and check global speed field on rank 0 ===
    lbm->computeSpeed();
    lbm->gatherDataOnRank0(LbmDQ::Speed);
    if (rank == 0) {
        float* speed = lbm->getGatheredSpeed();
        int N = lbm->getTotalDimX() * lbm->getTotalDimY() * lbm->getTotalDimZ();
        float min_prop = 1e10, max_prop = -1e10, sum_prop = 0.0;
	for (int i = 0; i < N; ++i) {
            float prop = speed[i];
            if (prop < min_prop) min_prop = prop;
            if (prop > max_prop) max_prop = prop;
            sum_prop += prop;
        }
        //printf("\nGlobal speed: min=%f, max=%f, mean=%f\n", min_prop, max_prop, sum_prop / N);
    }
// === END global vorticity check ===

    // Gather speed data for export (rank 0 only)
    lbm->computeSpeed();
    lbm->gatherDataOnRank0(LbmDQ::Speed);

    // Export simulation state to file (rank 0 only, after gathering)
    if (rank == 0) {
        exportSimulationStateToFile(lbm, "speed_state60.txt");
	exportSimulationStateToVTK(lbm, "sim_sphere_speed60.vtk");
    }

    // Print last 5 values and guard bytes of dbl_arrays for debugging
    int Q = lbm->getQ();
    uint32_t size = lbm->getDimX() * lbm->getDimY() * lbm->getDimZ();
    float* dbl_arrays = lbm->getDblArrays();
//    std::cout << "[Rank " << rank << "] DEBUG: dim_x=" << lbm->getDimX() << ", dim_y=" << lbm->getDimY() << ", dim_z=" << lbm->getDimZ() 
//              << ", size=" << size << ", total_allocated=" << (Q+6)*size + LbmDQ::GUARD_SIZE 
//              << ", density_offset=" << Q*size << ", speed_offset=" << (Q+5)*size << std::endl;
    for (int i = (Q+6)*size - 5; i < (int)((Q+6)*size + LbmDQ::GUARD_SIZE); ++i) {
//        std::cout << "[Rank " << rank << "] dbl_arrays[" << i << "] = " << dbl_arrays[i] << std::endl;
    }

    // === Print time series for near-barrier nodes after timing summary ===
    if (rank < MAX_PRINT_RANK && !near_barrier_nodes.empty()) {
        for (size_t n = 0; n < near_barrier_nodes.size(); ++n) {
            const auto& probe = near_barrier_nodes[n];
            printf("\n[END] Rank %d near-barrier node %zu (global %d,%d,%d | local %d,%d,%d):\n", rank, n, probe.global_X, probe.global_Y, probe.global_Z, probe.local_i, probe.local_j, probe.local_k);
            for (size_t s = 0; s < probe.steps.size(); ++s) {
                printf("[t=%d] velocity=(%.5f,%.5f,%.5f) density=%.5f\n", probe.steps[s], probe.vx[s], probe.vy[s], probe.vz[s], probe.density[s]);
            }
        }
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

// Export simulation state to file: one line per element (x y z value)
void exportSimulationStateToFile(LbmDQ* lbm, const char* filename) {
    // Choose which scalar to export by changing this line:
    float* gathered = lbm->getGatheredSpeed(); // <-- Change to getGatheredVorticity() or getGatheredDensity() as needed
    if (!gathered) {
        fprintf(stderr, "Error: gathered array is NULL!\n");
        return;
    }
    const char* scalar_name = "speed_m_per_s"; // <-- Change to "vorticity" or "density" to match above
    double value_scale = 1.0; // <-- Will be set below

    // Physical parameters (should match those in runLbmCfdSimulation)
    double physical_density = 1380.0; // kg/m^3
    double physical_length = 2.0; // meters
    double physical_time = 8.0;   // seconds
    int time_steps = 20000;
    int dim_y = 60;
    double dt = physical_time / time_steps;
    double dx = physical_length / (double)dim_y;
    double speed_scale = dx / dt; // lattice to m/s

    // Set scaling based on scalar_name
    if (std::string(scalar_name) == "speed_m_per_s") {
        value_scale = speed_scale;
    } else if (std::string(scalar_name) == "density") {
        value_scale = physical_density; // lattice density is 1.0 -> physical_density
    } else if (std::string(scalar_name) == "vorticity") {
        value_scale = speed_scale / dx; // lattice vorticity (1/timestep) -> 1/s
    }

    int total_x = lbm->getTotalDimX();
    int total_y = lbm->getTotalDimY();
    int total_z = lbm->getTotalDimZ();
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s for writing\n", filename);
        return;
    }
    // Write header
    fprintf(fp, "# x y z %s\n", scalar_name);
    for (int k = 0; k < total_z; ++k) {
        for (int j = 0; j < total_y; ++j) {
            for (int i = 0; i < total_x; ++i) {
                int idx = i + total_x * (j + total_y * k);
                float value = gathered[idx] * value_scale;
                fprintf(fp, "%d %d %d %.6f\n", i, j, k, value);
            }
        }
    }
    fclose(fp);
    printf("Export completed successfully\n");
}

void exportSimulationStateToVTK(LbmDQ* lbm, const char* filename) {
    // Choose which scalar to export by changing this line:
    float* gathered = lbm->getGatheredSpeed(); // <-- Change to getGatheredVorticity() or getGatheredDensity() as needed
    if (!gathered) {
        fprintf(stderr, "Error: gathered array is NULL!\n");
        return;
    }
    const char* scalar_name = "speed_m_per_s"; // <-- Change to "vorticity" or "density" or "speed_m_per_s" to match above
    double value_scale = 1.0; // <-- Will be set below

    // Physical parameters (should match those in runLbmCfdSimulation)
    double physical_density = 1380.0; // kg/m^3
    double physical_length = 2.0; // meters
    double physical_time = 8.0;   // seconds
    int time_steps = 20000;
    int dim_y = 60;
    double dt = physical_time / time_steps;
    double dx = physical_length / (double)dim_y;
    double speed_scale = dx / dt; // lattice to m/s

    // Set scaling based on scalar_name
    if (std::string(scalar_name) == "speed_m_per_s") {
        value_scale = speed_scale;
    } else if (std::string(scalar_name) == "density") {
        value_scale = physical_density; // lattice density is 1.0 -> physical_density
    } else if (std::string(scalar_name) == "vorticity") {
        value_scale = speed_scale / dx; // lattice vorticity (1/timestep) -> 1/s
    }

    int total_x = lbm->getTotalDimX();
    int total_y = lbm->getTotalDimY();
    int total_z = lbm->getTotalDimZ();
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s for writing\n", filename);
        return;
    }
    // Write VTK header
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "LBM Simulation Data (%s)\n", scalar_name);
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_GRID\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", total_x, total_y, total_z);
    fprintf(fp, "POINTS %d float\n", total_x * total_y * total_z);
    // Write coordinates
    for (int k = 0; k < total_z; ++k) {
        for (int j = 0; j < total_y; ++j) {
            for (int i = 0; i < total_x; ++i) {
                fprintf(fp, "%d %d %d\n", i, j, k);
            }
        }
    }
    // Write scalar data
    fprintf(fp, "POINT_DATA %d\n", total_x * total_y * total_z);
    fprintf(fp, "SCALARS %s float 1\n", scalar_name);
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k = 0; k < total_z; ++k) {
        for (int j = 0; j < total_y; ++j) {
            for (int i = 0; i < total_x; ++i) {
                int idx = i + total_x * (j + total_y * k);
                fprintf(fp, "%.6f\n", gathered[idx] * value_scale);
            }
        }
    }
    fclose(fp);
    printf("VTK export completed: %s\n", filename);
}
