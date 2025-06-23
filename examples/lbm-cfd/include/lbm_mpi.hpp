#ifndef _LBMDQ_MPI_HPP_
#define _LBMDQ_MPI_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <vector>

#define MPI_CHECK(call) \
    do { \
        int mpi_err = (call); \
        if (mpi_err != MPI_SUCCESS) { \
            char err_str[MPI_MAX_ERROR_STRING]; \
            int err_len; \
            MPI_Error_string(mpi_err, err_str, &err_len); \
            std::cerr << "MPI error at " << __FILE__ << ":" << __LINE__ << ": " << err_str << std::endl; \
            MPI_Abort(MPI_COMM_WORLD, mpi_err); \
        } \
    } while (0)

// Add bounds checking macro
#ifndef LBM_BOUNDS_CHECK
#define LBM_BOUNDS_CHECK(idx, arrsize, msg) \
    if ((idx) < 0 || (idx) >= (arrsize)) { \
        std::cerr << "Out-of-bounds access: " << (msg) << " at idx=" << (idx) << ", arrsize=" << (arrsize) << std::endl; \
	MPI_Abort(MPI_COMM_WORLD, 1); \
    }
#endif

// Helper class for creating barriers
class Barrier
{
    public:
        enum Type {HORIZONTAL, VERTICAL};
    protected:
        Type type;
        int x1;
        int x2;
        int y1;
        int y2;

    public:
        Type getType() { return type; }
        int getX1() { return x1; }
        int getX2() { return x2; }
        int getY1() { return y1; }
        int getY2() { return y2; }
};

class BarrierHorizontal : public Barrier
{
    public:
        BarrierHorizontal(int x_start, int x_end, int y) {
            type = Barrier::HORIZONTAL;
            x1 = x_start;
            x2 = x_end;
            y1 = y;
            y2 = y;
        }
        ~BarrierHorizontal() {}
};

class BarrierVertical : public Barrier
{
    public:
        BarrierVertical(int y_start, int y_end, int x) {
            type = Barrier::VERTICAL;
            x1 = x;
            x2 = x;
            y1 = y_start;
            y2 = y_end;
        }
        ~BarrierVertical() {}
};

// D3Q15 discrete velocities and weights
static const int cD3Q15[15][3] = {
	{0,0,0}, {1,0,0}, {-1,0,0},
	{0,1,0}, {0,-1,0}, {0,0,1},
	{0,0,-1}, {1,1,1}, {-1,1,1},
	{1,-1,1}, {1,1,-1}, {-1,-1,1},
	{-1,1,-1}, {1,-1,-1}, {-1,-1,-1}
};

static const double wD3Q15[15] = {
	2.0/9.0,
	1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
	1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0
};

// D3Q19 discrete velocities and weights
static const int cD3Q19[19][3] = {
    {0,0,0}, {1,0,0}, {-1,0,0},
    {0,1,0}, {0,-1,0}, {0,0,1},
    {0,0,-1}, {1,1,0}, {-1,1,0},
    {1,-1,0}, {-1,-1,0}, {1,0,1},
    {-1,0,1}, {1,0,-1}, {-1,0,-1},
    {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1}
};

static const double wD3Q19[19] = {
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// D3Q27 discrete velocities and weights
static const int cD3Q27[27][3] = {
    {0,0,0},
    {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
    {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0},
    {1,0,1}, {-1,0,1}, {1,0,-1}, {-1,0,-1},
    {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1},
    {1,1,1}, {-1,1,1}, {1,-1,1}, {-1,-1,1},
    {1,1,-1}, {-1,1,-1}, {1,-1,-1}, {-1,-1,-1}
};

static const double wD3Q27[27] = {
    8.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};

// Lattice-Boltzman Methods CFD simulation
class LbmDQ
{
    public:
        enum FluidProperty {None, Density, Speed, Vorticity};
	enum LatticeType {D3Q15, D3Q19, D3Q27};

	// Guard byte constants
	static constexpr int GUARD_SIZE = 8;
        static constexpr float GUARD_FLOAT = 1e30f;
        static constexpr bool GUARD_BOOL = (bool)0xAB;


    private:
        enum Neighbor {NeighborN, NeighborE, NeighborS, NeighborW, NeighborNE, NeighborNW, NeighborSE, NeighborSW, NeighborUp, NeighborDown};
        enum Column {LeftBoundaryCol, LeftCol, RightCol, RightBoundaryCol};

        int rank;
        int num_ranks;
        uint32_t total_x;
        uint32_t total_y;
	uint32_t total_z;
        uint32_t dim_x;
        uint32_t dim_y;
	uint32_t dim_z;
        uint32_t start_x;
        uint32_t start_y;
	uint32_t start_z;
        uint32_t num_x;
        uint32_t num_y;
	uint32_t num_z;
        int offset_x;
        int offset_y;
	int offset_z;
        uint32_t *rank_local_size;
        uint32_t *rank_local_start;
        double speed_scale;
        int Q;
	LatticeType lattice_type;
	const int (*c)[3];
	const double *w;
	float *f;
        float *density;
        float *velocity_x;
        float *velocity_y;
	float *velocity_z;
        float *vorticity;
        float *speed;
        bool *barrier;
        FluidProperty stored_property;
        float *recv_buf;
        bool *brecv_buf;
        int neighbors[10];
        MPI_Datatype columns_2d[4];
        MPI_Datatype own_scalar;
        MPI_Datatype own_bool;
        MPI_Datatype *other_scalar;
        MPI_Datatype *other_bool;

	MPI_Comm cart_comm;
	MPI_Datatype faceXlo, faceXhi;
	MPI_Datatype faceYlo, faceYhi;
	MPI_Datatype faceZlo, faceZhi;
	MPI_Datatype faceN, faceS;
	MPI_Datatype faceE, faceW;
	MPI_Datatype faceSW, faceNE;
	MPI_Datatype faceNW, faceSE;

        // Add missing member variables
        float *f_0, *f_1, *f_2, *f_3, *f_4, *f_5, *f_6, *f_7, *f_8, *f_9, *f_10, *f_11, *f_12, *f_13, *f_14, *f_15, *f_16, *f_17, *f_18, *f_19, *f_20, *f_21, *f_22, *f_23, *f_24, *f_25, *f_26;
        float *dbl_arrays;
        uint32_t block_width, block_height, block_depth;
	float **fPtr;
	uint32_t array_size;
	
	// Temporary storage for bounce-back streaming
	std::vector<float> f_Old;
	size_t last_size;

	// Helper function to print memory usage
        void printMemoryUsage(const char* label, size_t bytes) {
            double mb = bytes / (1024.0 * 1024.0);
            double gb = mb / 1024.0; // NOLINT
            if (rank == 0) {
            //std::cout << "Memory usage - " << label << ": " << mb << " MB (" << gb << " GB)" << std::endl;
            }
        }

	// Helper functions
        inline int idx3D(int x, int y, int z) const {
            if (x < 0 || x >= dim_x || y < 0 || y >= dim_y || z < 0 || z >= dim_z) {
                std::cerr << "[Rank " << rank << "] ERROR: idx3D out of bounds: x=" << x << ", y=" << y << ", z=" << z 
                          << ", dim_x=" << dim_x << ", dim_y=" << dim_y << ", dim_z=" << dim_z << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
	    return x + dim_x * (y + dim_y * z);
        }

        inline float& f_at(int d, int x, int y, int z) const {
            int idx = idx3D(x, y, z);
            LBM_BOUNDS_CHECK(d, Q, "f_at direction");
            LBM_BOUNDS_CHECK(idx, dim_x * dim_y * dim_z, "f_at index");
	    return fPtr[d][idx];
        }

        void setEquilibrium(int x, int y, int z, double new_velocity_x, double new_velocity_y, double new_velocity_z, double new_density);
        void getClosestFactors3(int value, int *factor_1, int *factor_2, int *factor_3);

    public:
        LbmDQ(uint32_t width, uint32_t height, uint32_t depth, double scale, int task_id, int num_tasks, LatticeType type);
        ~LbmDQ();

        void initBarrier(std::vector<Barrier*> barriers);
        void initFluid(double physical_speed);
        void updateFluid(double physical_speed);
        void collide(double viscosity);
        void stream();
        void bounceBackStream();
        bool checkStability();
        void computeSpeed();
        void computeVorticity();
        void gatherDataOnRank0(FluidProperty property);
        void exchangeBoundaries();
	uint32_t getDimX();
        uint32_t getDimY();
	uint32_t getDimZ();
        uint32_t getTotalDimX();
        uint32_t getTotalDimY();
	uint32_t getTotalDimZ();
        uint32_t getOffsetX();
        uint32_t getOffsetY();
	uint32_t getOffsetZ();
        uint32_t getStartX();
        uint32_t getStartY();
	uint32_t getStartZ();
        uint32_t getSizeX();
        uint32_t getSizeY();
	uint32_t getSizeZ();
        uint32_t* getRankLocalSize(int rank);
        uint32_t* getRankLocalStart(int rank);
        bool* getBarrier();
	static constexpr int TAG_F  = 100;
        static constexpr int TAG_D  = 101;
        static constexpr int TAG_VX = 102;
        static constexpr int TAG_VY = 103;
        static constexpr int TAG_VZ = 104;
	static constexpr int TAG_B  = 105;

	// Local data access
        inline float* getDensity() {
            return density;
        }

        inline float* getVelocityX() {
            return velocity_x;
        }

        inline float* getVelocityY() {
            return velocity_y;
        }

        inline float* getVelocityZ() {
            return velocity_z;
        }

        inline float* getVorticity() {
            return vorticity;
        }

        inline float* getSpeed() {
            return speed;
        }

        // Gathered data access (rank 0 only)
        inline float* getGatheredDensity() {
            if (rank != 0 || stored_property != Density) return NULL;
            return recv_buf;
        }

        inline float* getGatheredVorticity() {
            if (rank != 0 || stored_property != Vorticity) return NULL;
            return recv_buf;
        }

        inline float* getGatheredSpeed() {
            if (rank != 0 || stored_property != Speed) return NULL;
            return recv_buf;
        }

	// Check guard bytes for buffer overruns
        void checkGuards() {
            bool guard_ok = true;
	    std::cout << "[Rank " << rank << "] checkGuards: recv_buf=" << static_cast<void*>(recv_buf) << ", array_size=" << array_size << std::endl;
            for (int i = 0; i < GUARD_SIZE; ++i) {
                if (recv_buf && recv_buf[array_size + i] != GUARD_FLOAT) {
                    std::cerr << "[Rank " << rank << "] WARNING: recv_buf guard byte " << i << " corrupted!" << std::endl;
                    guard_ok = false;
                }
	    }
            std::cout << "[Rank " << rank << "] checkGuards: brecv_buf=" << static_cast<void*>(brecv_buf) << ", array_size=" << array_size << std::endl;
            for (int i = 0; i < GUARD_SIZE; ++i) {
                if (brecv_buf && brecv_buf[array_size + i] != GUARD_BOOL) {
                    std::cerr << "[Rank " << rank << "] WARNING: brecv_buf guard byte " << i << " corrupted!" << std::endl;
                    guard_ok = false;
                }
            }
            uint32_t size = dim_x * dim_y * dim_z;
            std::cout << "[Rank " << rank << "] checkGuards: dbl_arrays=" << static_cast<void*>(dbl_arrays) << ", (Q+6)*size=" << (Q+6)*size << std::endl;
	    if (dbl_arrays) {
                uint32_t dbl_arrays_size = dim_x * dim_y * dim_z;
		for (int i = 0; i < GUARD_SIZE; ++i) {
                    if (dbl_arrays[(Q + 6) * dbl_arrays_size + i] != GUARD_FLOAT) {
                        std::cerr << "[Rank " << rank << "] WARNING: dbl_arrays guard byte " << i << " corrupted!" << std::endl;
                        guard_ok = false;
                    }
                }
            }
            if (!guard_ok) {
                std::cerr << "[Rank " << rank << "] WARNING: Buffer overrun detected in checkGuards!" << std::endl;
            } else {
                std::cout << "[Rank " << rank << "] checkGuards: All guard bytes OK." << std::endl;
            }
        }
};

// constructor
LbmDQ::LbmDQ(uint32_t width, uint32_t height, uint32_t depth, double scale, int task_id, int num_tasks, LatticeType type)
{
    rank = task_id;
    num_ranks = num_tasks;
    speed_scale = scale;
    stored_property = None;
    lattice_type = type;

    // set up lattice model
    if (lattice_type == D3Q15) {
	Q = 15;
	c = cD3Q15;
	w = wD3Q15;
    }
    else if (lattice_type == D3Q19) {
	Q = 19;
	c = cD3Q19;
	w = wD3Q19;
    }
    else if (lattice_type == D3Q27) {
	Q = 27;
	c = cD3Q27;
	w = wD3Q27;
    }

    // split up problem space
    int n_x, n_y, n_z, col, row, layer, chunk_w, chunk_h, chunk_d, extra_w, extra_h, extra_d;
    getClosestFactors3(num_ranks, &n_x, &n_y, &n_z);
    chunk_w = width / n_x;
    chunk_h = height / n_y;
    chunk_d = depth / n_z;
    extra_w = width % n_x;
    extra_h = height % n_y;
    extra_d = depth % n_z;
    col = rank % n_x;
    row = (rank / n_x) % n_y;
    layer = rank / (n_x * n_y);

    num_x = chunk_w + ((col < extra_w) ? 1 : 0);
    num_y = chunk_h + ((row < extra_h) ? 1 : 0);
    num_z = chunk_d + ((layer < extra_d) ? 1 : 0);
    offset_x = col * chunk_w + std::min<int>(col, extra_w);
    offset_y = row * chunk_h + std::min<int>(row, extra_h);
    offset_z = layer * chunk_d + std::min<int>(layer, extra_d);
    start_x = (col == 0) ? 0 : 1;
    start_y = (row == 0) ? 0 : 1;
    start_z = (layer == 0) ? 0 : 1;

    // print subdomain info for all ranks
    if (rank == 0) {
        std::cout << "Partitioning: n_x=" << n_x << ", n_y=" << n_y << ", n_z=" << n_z << std::endl;
    }                                                                           std::cout << "[Rank " << rank << "] num_x=" << num_x << ", num_y=" << num_y << ", num_z=" << num_z
              << ", offset_x=" << offset_x << ", offset_y=" << offset_y << ", offset_z=" << offset_z << std::endl;

    // set up sub grid for simulation
    total_x = width;
    total_y = height;
    total_z = depth;
    block_width = num_x;
    block_height = num_y;
    block_depth = num_z;
    dim_x = block_width;
    dim_y = block_height;
    dim_z = block_depth;

    // Set array_size before allocation
    if (rank == 0) {
        array_size = total_x * total_y * total_z;
    } else {
        array_size = dim_x * dim_y * dim_z;
    }

    // Debug: print before allocations
//    std::cout << "[Rank " << rank << "] About to allocate recv_buf/brecv_buf, array_size=" << array_size << std::endl;

    // Allocate recv_buf and brecv_buf with guard bytes
    recv_buf = new float[array_size + GUARD_SIZE];
    brecv_buf = new bool[array_size + GUARD_SIZE];
    // Set guard values
    for (int i = 0; i < GUARD_SIZE; ++i) {
        recv_buf[array_size + i] = GUARD_FLOAT;
        brecv_buf[array_size + i] = GUARD_BOOL;
    }
//    std::cout << "[Rank " << rank << "] Finished allocating recv_buf/brecv_buf" << std::endl;

    // MPI Checks - Abort 
    if (num_x == 0 || num_y == 0 || num_z == 0) {
    std::cerr << "[Rank " << rank << "] ERROR: Subdomain has zero size! num_x=" << num_x << ", num_y=" << num_y << ", num_z=" << num_z << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (offset_x < 0 || offset_y < 0 || offset_z < 0) {
        std::cerr << "[Rank " << rank << "] ERROR: Negative offsets! offset_x=" << offset_x << ", offset_y=" << offset_y << ", offset_z=" << offset_z << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (offset_x + num_x > width || offset_y + num_y > height || offset_z + num_z > depth) {
        std::cerr << "[Rank " << rank << "] ERROR: Subdomain extends beyond global domain!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    neighbors[NeighborN] = (row == n_y-1) ? MPI_PROC_NULL : rank + n_x;
    neighbors[NeighborE] = (col == n_x-1) ? MPI_PROC_NULL : rank + 1;
    neighbors[NeighborS] = (row == 0) ? MPI_PROC_NULL : rank - n_x; 
    neighbors[NeighborW] = (col == 0) ? MPI_PROC_NULL : rank - 1;
    neighbors[NeighborNE] = (row == n_y-1 || col == n_x-1) ? MPI_PROC_NULL : rank + n_x + 1;
    neighbors[NeighborNW] = (row == n_y-1 || col == 0) ? MPI_PROC_NULL : rank + n_x - 1;
    neighbors[NeighborSE] = (row == 0 || col == n_x-1) ? MPI_PROC_NULL : rank - n_x + 1;
    neighbors[NeighborSW] = (row == 0 || col == 0) ? MPI_PROC_NULL : rank - n_x - 1;
    
    //new Z neighbors
    neighbors[NeighborUp] = (layer == n_z-1) ? MPI_PROC_NULL : rank + (n_x * n_y);
    neighbors[NeighborDown] = (layer == 0) ? MPI_PROC_NULL : rank - (n_x * n_y);
    
    int dims[3] = {n_z, n_y, n_x};
    int periods[3] = {0, 0, 0};
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);

    // create data types for exchanging data with neighbors
    int sizes3D[3] = {int(total_z), int(total_y), int(total_x)};
    int subsize3D[3] = {int(num_z), int(num_y), int(num_x)};
    int offsets3D[3] = {int(offset_z), int(offset_y), int(offset_x)};

    // Debug: own_scalar
//    std::cout << "[Rank " << rank << "] Creating MPI datatype own_scalar: sizes=" << sizes3D[0] << "," << sizes3D[1] << "," << sizes3D[2]
//              << " subsize=" << subsize3D[0] << "," << subsize3D[1] << "," << subsize3D[2]
//              << " offsets=" << offsets3D[0] << "," << offsets3D[1] << "," << offsets3D[2] << std::endl;

    MPI_Type_create_subarray(3, sizes3D, subsize3D, offsets3D, MPI_ORDER_C, MPI_DOUBLE, &own_scalar);
    MPI_Type_commit(&own_scalar);

//    std::cout << "[Rank " << rank << "] Finished creating MPI datatype own_scalar" << std::endl;

//    std::cout << "[Rank " << rank << "] Creating MPI datatype own_bool: sizes=" << sizes3D[0] << "," << sizes3D[1] << "," << sizes3D[2]
//              << " subsize=" << subsize3D[0] << "," << subsize3D[1] << "," << subsize3D[2]
//              << " offsets=" << offsets3D[0] << "," << offsets3D[1] << "," << offsets3D[2] << std::endl;

    MPI_Type_create_subarray(3, sizes3D, subsize3D, offsets3D, MPI_ORDER_C, MPI_BYTE, &own_bool);
    MPI_Type_commit(&own_bool);

//    std::cout << "[Rank " << rank << "] Finished creating MPI datatype own_bool" << std::endl;

    other_scalar = new MPI_Datatype[num_ranks];
    other_bool = new MPI_Datatype[num_ranks];

    for (int r = 0; r < num_ranks; r++) {
        int col = r % n_x;
        int row = (r / n_x) % n_y;
        int layer = r / (n_x * n_y);
        int osub[3] = {int(chunk_d + (layer < extra_d)), int(chunk_h + (row < extra_h)), int(chunk_w + (col < extra_w))};
        int ooffset[3] = {int(layer * chunk_d + std::min(layer, extra_d)), 
                         int(row * chunk_h + std::min(row, extra_h)),
                         int(col * chunk_w + std::min(col, extra_w))};

//	std::cout << "[Rank " << rank << "] Creating MPI datatype other_scalar[r=" << r << "]: sizes=" << sizes3D[0] << "," << sizes3D[1] << "," << sizes3D[2]
//                  << " subsize=" << osub[0] << "," << osub[1] << "," << osub[2]
//                  << " offsets=" << ooffset[0] << "," << ooffset[1] << "," << ooffset[2] << std::endl;

        MPI_Type_create_subarray(3, sizes3D, osub, ooffset, MPI_ORDER_C, MPI_DOUBLE, &other_scalar[r]);
        MPI_Type_commit(&other_scalar[r]);

//	std::cout << "[Rank " << rank << "] Finished creating MPI datatype other_scalar[r=" << r << "]" << std::endl;
//        std::cout << "[Rank " << rank << "] Creating MPI datatype other_bool[r=" << r << "]: sizes=" << sizes3D[0] << "," << sizes3D[1] << "," << sizes3D[2]
//                  << " subsize=" << osub[0] << "," << osub[1] << "," << osub[2]
//                  << " offsets=" << ooffset[0] << "," << ooffset[1] << "," << ooffset[2] << std::endl;

        MPI_Type_create_subarray(3, sizes3D, osub, ooffset, MPI_ORDER_C, MPI_BYTE, &other_bool[r]);
        MPI_Type_commit(&other_bool[r]);
//        std::cout << "[Rank " << rank << "] Finished creating MPI datatype other_bool[r=" << r << "]" << std::endl;	
    }

    //X-Faces
    int subsX[3]   = {int(num_z), int(num_y), 1};
    int offsXlo[3] = {int(start_z), int(start_y), int(start_x)};
    int offsXhi[3] = {int(start_z), int(start_y), int(dim_x - start_x - 1)};
    MPI_Type_create_subarray(3, sizes3D, subsX, offsXlo, MPI_ORDER_C, MPI_DOUBLE, &faceXlo);
    MPI_Type_create_subarray(3, sizes3D, subsX, offsXhi, MPI_ORDER_C, MPI_DOUBLE, &faceXhi);
    MPI_Type_commit(&faceXlo);
    MPI_Type_commit(&faceXhi);

    //Y-Faces
    int subsY[3]   = {int(num_z), 1, int(num_x)};
    int offsYlo[3] = {int(start_z), int(start_y), int(start_x)};
    int offsYhi[3] = {int(start_z), int(dim_y - start_y - 1), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsY, offsYlo, MPI_ORDER_C, MPI_DOUBLE, &faceYlo);
    MPI_Type_create_subarray(3, sizes3D, subsY, offsYhi, MPI_ORDER_C, MPI_DOUBLE, &faceYhi);
    MPI_Type_commit(&faceYlo);
    MPI_Type_commit(&faceYhi);

    //Z-Faces
    int subsZ[3]   = {1, int(num_y), int(num_x)};
    int offsZlo[3] = {int(start_z), int(start_y), int(start_x)};
    int offsZhi[3] = {int(dim_z - start_z - 1), int(start_y), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsZ, offsZlo, MPI_ORDER_C, MPI_DOUBLE, &faceZlo);
    MPI_Type_create_subarray(3, sizes3D, subsZ, offsZhi, MPI_ORDER_C, MPI_DOUBLE, &faceZhi);
    MPI_Type_commit(&faceZlo);
    MPI_Type_commit(&faceZhi);


    //North
    int subsN[3]   = {int(num_z), 1, int(num_x)};
    int offsN[3]   = {int(start_z), int(dim_y - start_y -1), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsN, offsN, MPI_ORDER_C, MPI_DOUBLE, &faceN);
    MPI_Type_commit(&faceN);

    //South
    int subsS[3]   = {int(num_z), 1, int(num_x)};
    int offsS[3]   = {int(start_z), int(start_y), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsS, offsS, MPI_ORDER_C, MPI_DOUBLE, &faceS);
    MPI_Type_commit(&faceS);

    //East
    int subsE[3]   = {int(num_z), int(num_y), 1};
    int offsE[3]   = {int(start_z), int(start_y), int(dim_x - start_x - 1)};
    MPI_Type_create_subarray(3, sizes3D, subsE, offsE, MPI_ORDER_C, MPI_DOUBLE, &faceE);
    MPI_Type_commit(&faceE);

    //West
    int subsW[3]   = {int(num_z), int(num_y), 1};
    int offsW[3]   = {int(start_z), int(start_y), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsW, offsW, MPI_ORDER_C, MPI_DOUBLE, &faceW);
    MPI_Type_commit(&faceW);

    //Northeast
    int subsNE[3]   = {int(num_z), 1, 1};
    int offsNE[3]   = {int(start_z), int(dim_y - start_y - 1), int(dim_x - start_x - 1)};
    MPI_Type_create_subarray(3, sizes3D, subsNE, offsNE, MPI_ORDER_C, MPI_DOUBLE, &faceNE);
    MPI_Type_commit(&faceNE);

    //Northwest
    int subsNW[3]   = {int(num_z), 1, 1};
    int offsNW[3]   = {int(start_z), int(dim_y - start_y - 1), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsNW, offsNW, MPI_ORDER_C, MPI_DOUBLE, &faceNW);
    MPI_Type_commit(&faceNW);

    //Southeast
    int subsSE[3]   = {int(num_z), 1, 1};
    int offsSE[3]   = {int(start_z), int(start_y), int(dim_x - start_x - 1)};
    MPI_Type_create_subarray(3, sizes3D, subsSE, offsSE, MPI_ORDER_C, MPI_DOUBLE, &faceSE);
    MPI_Type_commit(&faceSE);

    //Southwest
    int subsSW[3]   = {int(num_z), 1, 1};
    int offsSW[3]   = {int(start_z), int(start_y), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsSW, offsSW, MPI_ORDER_C, MPI_DOUBLE, &faceSW);
    MPI_Type_commit(&faceSW);

    // Number of guard elements
    constexpr int GUARD_SIZE = 8;
    constexpr float GUARD_FLOAT = 1e30f;
    constexpr bool GUARD_BOOL = (bool)0xAB;

    // Allocate recv_buf and brecv_buf with guard bytes
    //if (rank == 0) {
    //    array_size = total_x * total_y * total_z;
    //    recv_buf = new float[array_size + GUARD_SIZE];
    //    brecv_buf = new bool[array_size + GUARD_SIZE];
    //    // Set guard values
    //    for (int i = 0; i < GUARD_SIZE; ++i) {
    //        recv_buf[array_size + i] = GUARD_FLOAT;
    //        brecv_buf[array_size + i] = GUARD_BOOL;
    //    }
    //} else {
    //    array_size = dim_x * dim_y * dim_z;
    //    recv_buf = new float[array_size + GUARD_SIZE];
    //    brecv_buf = new bool[array_size + GUARD_SIZE];
    //    for (int i = 0; i < GUARD_SIZE; ++i) {
    //        recv_buf[array_size + i] = GUARD_FLOAT;
    //        brecv_buf[array_size + i] = GUARD_BOOL;
    //    }
    //}

    // Print MPI datatype sizes for debugging
    int size_own_scalar = 0, size_own_bool = 0;
    MPI_Type_size(own_scalar, &size_own_scalar);
//    MPI_Type_size(own_bool, &size_own_bool);
//    std::cout << "[Rank " << rank << "] MPI_Type_size(own_scalar): " << size_own_scalar << ", own_bool: " << size_own_bool << std::endl;
    for (int r = 0; r < num_ranks; r++) {
        int size_other_scalar = 0, size_other_bool = 0;
        MPI_Type_size(other_scalar[r], &size_other_scalar);
        MPI_Type_size(other_bool[r], &size_other_bool);
//        std::cout << "[Rank " << rank << "] MPI_Type_size(other_scalar[" << r << "]): " << size_other_scalar
//                  << ", other_bool[" << r << "]: " << size_other_bool << std::endl;
        if (size_other_scalar > (int)((total_x * total_y * total_z) * sizeof(float))) {
//            std::cerr << "[Rank " << rank << "] ERROR: other_scalar[" << r << "] type size (" << size_other_scalar << ") exceeds recv_buf allocation (" << (total_x * total_y * total_z) * sizeof(float) << ")!" << std::endl;
        }
        if (size_other_bool > (int)((total_x * total_y * total_z) * sizeof(bool))) {
//            std::cerr << "[Rank " << rank << "] ERROR: other_bool[" << r << "] type size (" << size_other_bool << ") exceeds brecv_buf allocation (" << (total_x * total_y * total_z) * sizeof(bool) << ")!" << std::endl;
        }
    }

    uint32_t size = dim_x * dim_y * dim_z;
    size_t total_memory = 0;

    // Debug: print before dbl_arrays allocation
//    std::cout << "[Rank " << rank << "] About to allocate dbl_arrays, size=" << (Q+6)*dim_x*dim_y*dim_z << std::endl;

    // allocate all float arrays at once
    dbl_arrays = new float[(Q + 6) * size + GUARD_SIZE];
    total_memory += (Q + 6) * size * sizeof(float);
    
    // Set guard values for dbl_arrays
    for (int i = 0; i < GUARD_SIZE; ++i) {
        dbl_arrays[(Q + 6) * size + i] = GUARD_FLOAT;
    }

//    std::cout << "[Rank " << rank << "] Finished allocating dbl_arrays" << std::endl;

    // set array pointers
    f_0        = dbl_arrays + (0*size);
    f_1        = dbl_arrays + (1*size);
    f_2        = dbl_arrays + (2*size);
    f_3        = dbl_arrays + (3*size);
    f_4        = dbl_arrays + (4*size);
    f_5        = dbl_arrays + (5*size);
    f_6        = dbl_arrays + (6*size);
    f_7        = dbl_arrays + (7*size);
    f_8        = dbl_arrays + (8*size);
    f_9        = dbl_arrays + (9*size);
    f_10       = dbl_arrays + (10*size);
    f_11       = dbl_arrays + (11*size);
    f_12       = dbl_arrays + (12*size);
    f_13       = dbl_arrays + (13*size);
    f_14       = dbl_arrays + (14*size);
    if (Q == 19) {
	f_15 = dbl_arrays + (15*size);
	f_16 = dbl_arrays + (16*size);
	f_17 = dbl_arrays + (17*size);
	f_18 = dbl_arrays + (18*size);
    }
    if (Q == 27) {
	f_15 = dbl_arrays + (15*size);
        f_16 = dbl_arrays + (16*size);
        f_17 = dbl_arrays + (17*size);
        f_18 = dbl_arrays + (18*size);
        f_19 = dbl_arrays + (19*size);
        f_20 = dbl_arrays + (20*size);
        f_21 = dbl_arrays + (21*size);
        f_22 = dbl_arrays + (22*size);
        f_23 = dbl_arrays + (23*size);
        f_24 = dbl_arrays + (24*size);
        f_25 = dbl_arrays + (25*size);
        f_26 = dbl_arrays + (26*size);
    }

    // initialize f pointer to point to f_0
    f = f_0;
    
    fPtr = new float*[Q];
    fPtr[0] = f_0;
    fPtr[1] = f_1;
    fPtr[2] = f_2;
    fPtr[3] = f_3;
    fPtr[4] = f_4;
    fPtr[5] = f_5;
    fPtr[6] = f_6;
    fPtr[7] = f_7;
    fPtr[8] = f_8;
    fPtr[9] = f_9;
    fPtr[10] = f_10;
    fPtr[11] = f_11;
    fPtr[12] = f_12;
    fPtr[13] = f_13;
    fPtr[14] = f_14;
    if (Q == 19) {
	fPtr[15] = f_15;
	fPtr[16] = f_16;
	fPtr[17] = f_17;
	fPtr[18] = f_18;
    }
    if (Q == 27) {
        fPtr[15] = f_15;
        fPtr[16] = f_16;
        fPtr[17] = f_17;
        fPtr[18] = f_18;
        fPtr[19] = f_19;
        fPtr[20] = f_20;
        fPtr[21] = f_21;
        fPtr[22] = f_22;
        fPtr[23] = f_23;
        fPtr[24] = f_24;
        fPtr[25] = f_25;
        fPtr[26] = f_26;
    }

    density    = dbl_arrays + (Q*size);
    velocity_x = dbl_arrays + ((Q+1)*size);
    velocity_y = dbl_arrays + ((Q+2)*size);
    velocity_z = dbl_arrays + ((Q+3)*size);
    vorticity  = dbl_arrays + ((Q+4)*size);
    speed      = dbl_arrays + ((Q+5)*size);
    
    // allocate boolean array
    barrier = new bool[size];
    total_memory += size * sizeof(bool);
    printMemoryUsage("Barrier array", size * sizeof(bool));

    // Allocate rank_local_size and rank_local_start arrays
    rank_local_size = new uint32_t[2 * num_ranks];
    rank_local_start = new uint32_t[2 * num_ranks];
    
    // Initialize the arrays with the correct values
    for (int r = 0; r < num_ranks; r++) {
        int col = r % n_x;
        int row = (r / n_x) % n_y;
        
        rank_local_size[2*r] = chunk_w + ((col < extra_w) ? 1 : 0);  // x size
        rank_local_size[2*r+1] = chunk_h + ((row < extra_h) ? 1 : 0); // y size
        
        rank_local_start[2*r] = col * chunk_w + std::min<int>(col, extra_w);  // x start
        rank_local_start[2*r+1] = row * chunk_h + std::min<int>(row, extra_h); // y start
    }

//    std::cout << "[Rank " << rank << "] Finished allocating recv_buf/brecv_buf" << std::endl;
}

// destructor
LbmDQ::~LbmDQ()
{
    std::cout << "[Rank " << rank << "] Entering destructor, array_size=" << array_size << std::endl;
    // Free MPI types
    if (faceXlo != MPI_DATATYPE_NULL) { MPI_Type_free(&faceXlo); faceXlo = MPI_DATATYPE_NULL; }
    if (faceXhi != MPI_DATATYPE_NULL) { MPI_Type_free(&faceXhi); faceXhi = MPI_DATATYPE_NULL; }
    if (faceYlo != MPI_DATATYPE_NULL) { MPI_Type_free(&faceYlo); faceYlo = MPI_DATATYPE_NULL; }
    if (faceYhi != MPI_DATATYPE_NULL) { MPI_Type_free(&faceYhi); faceYhi = MPI_DATATYPE_NULL; }
    if (faceZlo != MPI_DATATYPE_NULL) { MPI_Type_free(&faceZlo); faceZlo = MPI_DATATYPE_NULL; }
    if (faceZhi != MPI_DATATYPE_NULL) { MPI_Type_free(&faceZhi); faceZhi = MPI_DATATYPE_NULL; }
    if (faceN != MPI_DATATYPE_NULL)   { MPI_Type_free(&faceN);   faceN = MPI_DATATYPE_NULL; }
    if (faceS != MPI_DATATYPE_NULL)   { MPI_Type_free(&faceS);   faceS = MPI_DATATYPE_NULL; }
    if (faceE != MPI_DATATYPE_NULL)   { MPI_Type_free(&faceE);   faceE = MPI_DATATYPE_NULL; }
    if (faceW != MPI_DATATYPE_NULL)   { MPI_Type_free(&faceW);   faceW = MPI_DATATYPE_NULL; }
    if (faceNE != MPI_DATATYPE_NULL)  { MPI_Type_free(&faceNE);  faceNE = MPI_DATATYPE_NULL; }
    if (faceNW != MPI_DATATYPE_NULL)  { MPI_Type_free(&faceNW);  faceNW = MPI_DATATYPE_NULL; }
    if (faceSE != MPI_DATATYPE_NULL)  { MPI_Type_free(&faceSE);  faceSE = MPI_DATATYPE_NULL; }
    if (faceSW != MPI_DATATYPE_NULL)  { MPI_Type_free(&faceSW);  faceSW = MPI_DATATYPE_NULL; }
    if (own_scalar != MPI_DATATYPE_NULL) { MPI_Type_free(&own_scalar); own_scalar = MPI_DATATYPE_NULL; }
    if (own_bool != MPI_DATATYPE_NULL)   { MPI_Type_free(&own_bool);   own_bool = MPI_DATATYPE_NULL; }

    if (other_scalar != nullptr) {
        for (int i=0; i < num_ranks; i++) {
            if (other_scalar[i] != MPI_DATATYPE_NULL) {
                MPI_Type_free(&other_scalar[i]);
                other_scalar[i] = MPI_DATATYPE_NULL;
            }
        }
//	std::cout << "[Rank " << rank << "] Deleting other_scalar: " << static_cast<void*>(other_scalar) << std::endl;
	delete[] other_scalar;
        other_scalar = nullptr;
    }

    if (other_bool != nullptr) {
        for (int i=0; i < num_ranks; i++) {
            if (other_bool[i] != MPI_DATATYPE_NULL) {
                MPI_Type_free(&other_bool[i]);
                other_bool[i] = MPI_DATATYPE_NULL;
            }
        }
//	std::cout << "[Rank " << rank << "] Deleting other_bool: " << static_cast<void*>(other_bool) << std::endl;
        delete[] other_bool;
        other_bool = nullptr;
    }
   
    // Debug print before guard check
    std::cout << "[Rank " << rank << "] About to check guard bytes, array_size=" << array_size << ", recv_buf=" << static_cast<void*>(recv_buf) << ", brecv_buf=" << static_cast<void*>(brecv_buf) << std::endl;
    bool guard_ok = true;
    for (int i = 0; i < GUARD_SIZE; ++i) {
        if (recv_buf && recv_buf[array_size + i] != GUARD_FLOAT) {
            std::cerr << "[Rank " << rank << "] WARNING: recv_buf guard byte " << i << " corrupted!" << std::endl;
            guard_ok = false;
        }
        if (brecv_buf && brecv_buf[array_size + i] != GUARD_BOOL) {
            std::cerr << "[Rank " << rank << "] WARNING: brecv_buf guard byte " << i << " corrupted!" << std::endl;
            guard_ok = false;
        }
    }
    if (dbl_arrays) {
	uint32_t dbl_arrays_size = dim_x * dim_y * dim_z;
        for (int i = 0; i < GUARD_SIZE; ++i) {
            if (dbl_arrays[(Q + 6) * dbl_arrays_size + i] != GUARD_FLOAT) {
                std::cerr << "[Rank " << rank << "] WARNING: dbl_arrays guard byte " << i << " corrupted!" << std::endl;
                guard_ok = false;
            }
        }
    }
    if (!guard_ok) {
        std::cerr << "[Rank " << rank << "] WARNING: Buffer overrun detected before delete!" << std::endl;
    }
    std::cout << "[Rank " << rank << "] Finished guard check" << std::endl;

    // Debug prints before each delete
//    if (rank_local_size != nullptr) { std::cout << "[Rank " << rank << "] Deleting rank_local_size: " << static_cast<void*>(rank_local_size) << std::endl; delete[] rank_local_size; rank_local_size = nullptr; }
//    if (rank_local_start != nullptr) { std::cout << "[Rank " << rank << "] Deleting rank_local_start: " << static_cast<void*>(rank_local_start) << std::endl; delete[] rank_local_start; rank_local_start = nullptr; }
//    if (barrier != nullptr) { std::cout << "[Rank " << rank << "] Deleting barrier: " << static_cast<void*>(barrier) << std::endl; delete[] barrier; barrier = nullptr; }
//    if (recv_buf != nullptr) { std::cout << "[Rank " << rank << "] Deleting recv_buf: " << static_cast<void*>(recv_buf) << std::endl; delete[] recv_buf; recv_buf = nullptr; }
//    if (brecv_buf != nullptr) { std::cout << "[Rank " << rank << "] Deleting brecv_buf: " << static_cast<void*>(brecv_buf) << std::endl; delete[] brecv_buf; brecv_buf = nullptr; }
//    if (fPtr != nullptr) { std::cout << "[Rank " << rank << "] Deleting fPtr: " << static_cast<void*>(fPtr) << std::endl; delete[] fPtr; fPtr = nullptr; }
//    if (dbl_arrays != nullptr) { std::cout << "[Rank " << rank << "] Deleting dbl_arrays: " << static_cast<void*>(dbl_arrays) << std::endl; delete[] dbl_arrays; dbl_arrays = nullptr; }
    std::cout << "[Rank " << rank << "] Exiting destructor" << std::endl;

}

// initialize barrier based on selected type
void LbmDQ::initBarrier(std::vector<Barrier*> barriers)
{
//    std::cout << "[Rank " << rank << "] Entering initBarrier" << std::endl;
    // clear barrier to all `false`
    memset(barrier, 0, dim_x * dim_y * dim_z * sizeof(bool));
//    std::cout << "[Rank " << rank << "] Barrier array size: " << (dim_x * dim_y * dim_z) << std::endl;
    // set barrier to `true` where horizontal or vertical barriers exist
    int i, j;
//    std::cout << "[Rank " << rank << "] Number of barriers: " << barriers.size() << std::endl;
    for (i = 0; i < barriers.size(); i++) {
//        std::cout << "[Rank " << rank << "] Processing barrier " << i << ", type=" << (barriers[i]->getType() == Barrier::HORIZONTAL ? "HORIZONTAL" : "VERTICAL") << std::endl;
        if (barriers[i]->getType() == Barrier::Type::HORIZONTAL) {
            int global_y = barriers[i]->getY1();
            int y = global_y - offset_y;
//            std::cout << "[Rank " << rank << "] HORIZONTAL: global_y=" << global_y << ", local_y=" << y << std::endl;
            if (y >= 0 && y < dim_y) {
                for (j = barriers[i]->getX1(); j <= barriers[i]->getX2(); j++) {
                    int x = j - offset_x;
                    if (x >= 0 && x < dim_x) {
                        for (int k = 0; k < dim_z; ++k) {
                            int idx = idx3D(x, y, k);
			    if (idx >= 0 && idx < dim_x * dim_y * dim_z) {
                                barrier[idx] = true;
			    }
                        }
                    }
                }
            }
        }
        else {                // Barrier::VERTICAL
            int global_x = barriers[i]->getX1();
            int x = global_x - offset_x;
//            std::cout << "[Rank " << rank << "] VERTICAL: global_x=" << global_x << ", local_x=" << x << std::endl;
            if (x >= 0 && x < dim_x) {
                for (j = barriers[i]->getY1(); j <= barriers[i]->getY2(); j++) {
                    int y = j - offset_y;
                    if (y >= 0 && y < dim_y) {
                        // extrude vertical line through every Z-layer
                        for (int k = 0; k < dim_z; ++k) {
                            int idx = idx3D(x, y, k);
			    if (idx >= 0 && idx < dim_x * dim_y * dim_z) {
                                barrier[idx] = true;
			    }
                        }
                    }
                }
            }
        }
    }
//    std::cout << "[Rank " << rank << "] Exiting initBarrier" << std::endl;
}

// initialize fluid
void LbmDQ::initFluid(double physical_speed)
{
//    std::cout << "[Rank " << rank << "] Entering initFluid" << std::endl;
    int i, j, k;
    double speed = speed_scale * physical_speed;
    for (k = 0; k < dim_z; k++)
        {
        for (j = 0; j < dim_y; j++)
        {
            for (i = 0; i < dim_x; i++)
            {
                setEquilibrium(i, j, k, speed, 0.0, 0.0, 1.0);
                vorticity[idx3D(i, j, k)] = 0.0;
            }
        }
	}
//    std::cout << "[Rank " << rank << "] Exiting initFluid" << std::endl;
}

void LbmDQ::updateFluid(double physical_speed)
{
    int i; int j; int k;
    double speed = speed_scale * physical_speed;

    for (k = 0; k < dim_z; k++)
    {
	for (i = 0; i < dim_x; i++)
    	{
        setEquilibrium(i, 0, k, speed, 0.0, 0.0, 1.0);
        setEquilibrium(i, dim_y - 1, k, speed, 0.0, 0.0, 1.0);
    	}
    }
    
    for (k = 0; k < dim_z; k++)
    {
        for (j = 0; j < dim_y; j++)
        {
        setEquilibrium(0, j, k, speed, 0.0, 0.0, 1.0);
        setEquilibrium(dim_x - 1, j, k, speed, 0.0, 0.0, 1.0);
        }
    }

    for (j = 0; j < dim_y - 1; j++)
    {
        for (i = 0; i < dim_x - 1; i++)
        {
        setEquilibrium(i, j, 0, speed, 0.0, 0.0, 1.0);
        setEquilibrium(i, j, dim_z - 1, speed, 0.0, 0.0, 1.0);
        }
    }
}

// particle collision
void LbmDQ::collide(double viscosity)
{
//	std::cout << "[Rank " << rank << "] Entering collide" << std::endl;
	int i, j, k, idx;
	double omega = 1.0 / (3.0 * viscosity + 0.5); //reciprocal of relaxation time
	int arrsize = dim_x * dim_y * dim_z;
	
	for (k = 1; k < dim_z -1; k++)
	{
		for (j = 1; j < dim_y - 1; j++)
		{
			for (i = 1; i < dim_x - 1; ++i)
			{
				idx = idx3D(i, j, k);
				LBM_BOUNDS_CHECK(idx, arrsize, "collide main");
				double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
				for (int d = 0; d < Q; ++d)
				{
					LBM_BOUNDS_CHECK(idx, arrsize, "collide fPtr");
					double fv = fPtr[d][idx];
					rho += fv;
					ux  += fv * c[d][0];
					uy  += fv * c[d][1];
					uz  += fv * c[d][2];
				}
				density[idx] = rho;
				ux /= rho; uy /= rho; uz /= rho;
				velocity_x[idx] = ux;
				velocity_y[idx] = uy;
				velocity_z[idx] = uz;

				double usqr = ux*ux + uy*uy + uz*uz;
				for (int d = 0; d < Q; ++d)
				{
					double cu = 3.0 * (c[d][0]*ux + c[d][1]*uy + c[d][2]*uz);
					double feq = w[d] * rho * (1.0 + cu + 0.5*cu*cu - 1.5*usqr);
					LBM_BOUNDS_CHECK(idx, arrsize, "collide fPtr update");
					fPtr[d][idx] += omega * (feq - fPtr[d][idx]);
				}
			}
		}
	}
//	std::cout << "[Rank " << rank << "] Exiting collide" << std::endl;
}
	
// Optimized particle streaming - eliminates memory allocation/deallocation
void LbmDQ::stream()
{
//	std::cout << "[Rank " << rank << "] Entering stream" << std::endl;
	size_t slice = static_cast<size_t>(dim_x) * dim_y * dim_z;
	std::vector<float> temp_buffer(slice);

	for (int d = 0; d < Q; ++d) {
		float* f_d = fPtr[d];
		
		std::memcpy(temp_buffer.data(), f_d, slice * sizeof(float));

		for (int k = 0; k < dim_z; ++k) {
			for (int j = 0; j < dim_y; ++j) {
				for (int i = 0; i < dim_x; ++i) {
					int idx = idx3D(i, j, k);
					LBM_BOUNDS_CHECK(idx, slice, "stream");
					int ni = i + c[d][0];
					int nj = j + c[d][1];
					int nk = k + c[d][2];
					// Bounds check for destination indices
                    			if (ni < 0 || ni >= dim_x || nj < 0 || nj >= dim_y || nk < 0 || nk >= dim_z) continue;
					int nidx = idx3D(ni, nj, nk);
					LBM_BOUNDS_CHECK(nidx, slice, "stream fPtr update");
					f_d[nidx] = temp_buffer[idx];
				}
			}
		}
	}
//	std::cout << "[Rank " << rank << "] Exiting stream" << std::endl;
}

// Optimized bounce-back streaming - eliminates memory allocation/deallocation
void LbmDQ::bounceBackStream()
{
//	std::cout << "[Rank " << rank << "] Entering bounceBackStream" << std::endl;
	// Use a static buffer to avoid repeated allocation/deallocation
	static std::vector<std::vector<float>> f_Old_per_rank;
	static std::vector<std::vector<int>> opposite_dir_per_rank;
	static std::vector<size_t> last_size_per_rank;
	
	// Ensure we have enough entries for all ranks
	if (f_Old_per_rank.size() <= static_cast<size_t>(rank)) {
		f_Old_per_rank.resize(rank + 1);
		opposite_dir_per_rank.resize(rank + 1);
		last_size_per_rank.resize(rank + 1);
	}
	
	size_t slice = static_cast<size_t>(dim_x) * dim_y * dim_z;
	size_t total_size = Q * slice;
	
	// Only resize if needed for this rank
	if (f_Old_per_rank[rank].size() != total_size) {
		f_Old_per_rank[rank].resize(total_size);
		last_size_per_rank[rank] = total_size;
	}
	
	// Copy current state
	for (int d = 0; d < Q; ++d) {
		std::memcpy(f_Old_per_rank[rank].data() + d * slice, fPtr[d], slice * sizeof(float));
	}

	// Pre-compute opposite directions for better performance (rank-specific)
	if (opposite_dir_per_rank[rank].size() != Q) {
		opposite_dir_per_rank[rank].resize(Q);    
	    // Initialize all to -1 (invalid)
		for (int d = 0; d < Q; ++d) {
			opposite_dir_per_rank[rank][d] = -1;
		}
		// Find opposite directions
		for (int d = 0; d < Q; ++d) {
			for (int dd = 0; dd < Q; ++dd) {
	    			if (c[dd][0] == -c[d][0] && c[dd][1] == -c[d][1] && c[dd][2] == -c[d][2]) {
					opposite_dir_per_rank[rank][d] = dd;
					break;
				}
			}
		}
	}

	// Optimized bounce-back with better cache locality
	for (int k = 0; k < dim_z; ++k) {
		for (int j = 0; j < dim_y; ++j) {
			for (int i = 0; i < dim_x; ++i) {
				int idx = idx3D(i, j, k);
				LBM_BOUNDS_CHECK(idx, slice, "bounceBackStream");
				for (int d = 1; d < Q; ++d) {
					int ni = i + c[d][0];
					int nj = j + c[d][1];
					int nk = k + c[d][2];
					if (ni < 0 || ni >= dim_x || nj < 0 || nj >= dim_y || nk < 0 || nk >= dim_z) continue;
					int nidx = idx3D(ni, nj, nk);
					LBM_BOUNDS_CHECK(nidx, slice, "bounceBackStream fPtr update");
					LBM_BOUNDS_CHECK(nidx, slice, "bounceBackStream barrier access");
					if (barrier[nidx]) {
						int od = opposite_dir_per_rank[rank][d];
						if (od >= 0 && od < Q) {
							size_t f_old_idx = od * slice + idx;
							LBM_BOUNDS_CHECK(f_old_idx, total_size, "bounceBackStream f_Old access");
							fPtr[d][idx] = f_Old_per_rank[rank][f_old_idx];
						}
					}
				}
			}
		}
	}
//	std::cout << "[Rank " << rank << "] Exiting bounceBackStream" << std::endl;
}

// check if simulation has become unstable (if so, more time steps are required)
bool LbmDQ::checkStability()
{
    int i, k, idx;
    bool stable = true;
    int j = dim_y / 2;
    for (k = 0; k < dim_z; k++)
    {
	for (i = 0; i < dim_x; i++)
	    {
	        idx = idx3D(i, j, k);
	    LBM_BOUNDS_CHECK(idx, dim_x * dim_y * dim_z, "checkStability");
		if (density[idx] <= 0)
		{
		    stable = false;
		}
	    }
    }
    return stable;
}

// compute speed (magnitude of velocity vector)
void LbmDQ::computeSpeed()
{
    int i, j, k, idx;
    for (k = 1; k < dim_z - 1; k++)
    {
	for (j = 1; j < dim_y - 1; j++)
	{
            for (i = 1; i < dim_x - 1; i++)
            {
		idx = idx3D(i, j, k);
		LBM_BOUNDS_CHECK(idx, dim_x * dim_y * dim_z, "computeSpeed");
		speed[idx] = sqrt(velocity_x[idx] * velocity_x[idx] + velocity_y[idx] * velocity_y[idx] + velocity_z[idx] * velocity_z[idx]);
	    }
	}
    }
}

// compute vorticity (rotational velocity)
void LbmDQ::computeVorticity()
{
    int i; int j; int k; int idx;

    for (k = 1; k < dim_z - 1; k++)
    {
	for (j = 1; j < dim_y -1; j++)
	{
	    for (i = 1; i < dim_x - 1; i++)
	    {
		idx = idx3D(i, j, k);
		LBM_BOUNDS_CHECK(idx, dim_x * dim_y * dim_z, "computeVorticity");

		// Bounds check all neighboring accesses
		int idx_jp1 = idx3D(i, j + 1, k);
		int idx_jm1 = idx3D(i, j - 1, k);
		int idx_kp1 = idx3D(i, j, k + 1);
		int idx_km1 = idx3D(i, j, k - 1);
		int idx_ip1 = idx3D(i + 1, j, k);
		int idx_im1 = idx3D(i - 1, j, k);

		LBM_BOUNDS_CHECK(idx_jp1, dim_x * dim_y * dim_z, "computeVorticity j+1");
		LBM_BOUNDS_CHECK(idx_jm1, dim_x * dim_y * dim_z, "computeVorticity j-1");
		LBM_BOUNDS_CHECK(idx_kp1, dim_x * dim_y * dim_z, "computeVorticity k+1");
		LBM_BOUNDS_CHECK(idx_km1, dim_x * dim_y * dim_z, "computeVorticity k-1");
		LBM_BOUNDS_CHECK(idx_ip1, dim_x * dim_y * dim_z, "computeVorticity i+1");
		LBM_BOUNDS_CHECK(idx_im1, dim_x * dim_y * dim_z, "computeVorticity i-1");

		double wx = (velocity_z[idx_jp1] - velocity_z[idx_jm1]) - (velocity_y[idx_kp1] - velocity_y[idx_km1]);

		double wy = (velocity_z[idx_kp1] - velocity_z[idx_km1]) - (velocity_y[idx_ip1] - velocity_y[idx_im1]);

		double wz = (velocity_z[idx_ip1] - velocity_z[idx_im1]) - (velocity_y[idx_jp1] - velocity_y[idx_jm1]);

		vorticity[idx] = sqrt(wx*wx + wy*wy + wz*wz);
	    }
	}
    }
}

// gather all data on rank 0
void LbmDQ::gatherDataOnRank0(FluidProperty property)
{
    float *send_buf = NULL;
    bool *bsend_buf = barrier;
    switch (property)
    {
        case Density:
            send_buf = density;
            break;
        case Speed:
            send_buf = speed;
            break;
        case Vorticity:
            send_buf = vorticity;
            break;
        case None:
            return;
    }

    MPI_Status status;

    if (rank == 0)
    {
	MPI_Sendrecv(send_buf,  1, own_scalar, rank, TAG_F, recv_buf,  1, other_scalar[rank], rank, TAG_F, cart_comm, &status);
	MPI_Sendrecv(bsend_buf, 1, own_bool,   rank, TAG_B, brecv_buf, 1, other_bool[rank],   rank, TAG_B, cart_comm, &status);

	for (int r = 1; r < num_ranks; r++)
	{
	    MPI_Recv(recv_buf, 1, other_scalar[r], r, TAG_F, cart_comm, &status);
	    MPI_Recv(brecv_buf,1, other_bool[r],   r, TAG_B, cart_comm, &status);
	}
    }
    else
    {
	MPI_Send(send_buf,    1, own_scalar, 0, TAG_F, cart_comm);
        MPI_Send(bsend_buf,   1, own_bool,   0, TAG_B, cart_comm);
    }

    stored_property = property;
}

// get width of sub-area this task owns (including ghost cells)
uint32_t LbmDQ::getDimX()
{
    return dim_x;
}

// get width of sub-area this task owns (including ghost cells)
uint32_t LbmDQ::getDimY()
{
    return dim_y;
}

// get width of sub-area this task owns (including ghost cells)
uint32_t LbmDQ::getDimZ()
{
    return dim_z;
}

// get width of total area of simulation
uint32_t LbmDQ::getTotalDimX()
{
    return total_x;
}

// get width of total area of simulation
uint32_t LbmDQ::getTotalDimY()
{
    return total_y;
}

// get width of total area of simulation
uint32_t LbmDQ::getTotalDimZ()
{
    return total_z;
}

// get x offset into overall domain where this sub-area esxists
uint32_t LbmDQ::getOffsetX()
{
    return offset_x;
}

// get y offset into overall domain where this sub-area esxists
uint32_t LbmDQ::getOffsetY()
{
    return offset_y;
}

// get z offset into overall domain where this sub-area esxists
uint32_t LbmDQ::getOffsetZ()
{
    return offset_z;
}

// get x start for valid data (0 if no ghost cell on left, 1 if there is a ghost cell on left)
uint32_t LbmDQ::getStartX()
{
    return start_x;
}

// get y start for valid data (0 if no ghost cell on top, 1 if there is a ghost cell on top)
uint32_t LbmDQ::getStartY()
{
    return start_y;
}

// get z start for valid data (0 if no ghost cell on top, 1 if there is a ghost cell on top)
uint32_t LbmDQ::getStartZ()
{
    return start_z;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmDQ::getSizeX()
{
    return num_x;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmDQ::getSizeY()
{
    return num_y;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmDQ::getSizeZ()
{
    return num_z;
}

// get the local width and height of a particular rank's data
uint32_t* LbmDQ::getRankLocalSize(int rank)
{
    return rank_local_size + (2 * rank);
}

// get the local x and y start of a particular rank's data
uint32_t* LbmDQ::getRankLocalStart(int rank)
{
    return rank_local_start + (2 * rank);
}

// get barrier array
bool* LbmDQ::getBarrier()
{
    if (rank != 0) return NULL;
    return brecv_buf;
}

// private - set fluid equalibrium
void LbmDQ::setEquilibrium(int x, int y, int z, double new_velocity_x, double new_velocity_y, double new_velocity_z, double new_density)
{
	int idx = idx3D(x, y, z);
	LBM_BOUNDS_CHECK(idx, dim_x * dim_y * dim_z, "setEquilibrium");

	density[idx] = new_density;
	velocity_x[idx] = new_velocity_x;
	velocity_y[idx] = new_velocity_y;
	velocity_z[idx] = new_velocity_z;

	double ux = new_velocity_x;
	double uy = new_velocity_y;
	double uz = new_velocity_z;
	double usq = ux*ux + uy*uy + uz*uz;

	for (int d = 0; d < Q; ++d)
	{
		double cu = 3.0 * (c[d][0] * ux + c[d][1] * uy + c[d][2] * uz);
		f_at(d, x, y, z) = w[d] * new_density * (1.0 + cu + 0.5*cu*cu - 1.5*usq);
	}
}

// private - get 3 factors of a given number that are closest to each other
void LbmDQ::getClosestFactors3(int value, int *factor_1, int *factor_2, int *factor_3)
{
    //int test_num = (int)cbrt(value);
    //while (test_num > 0 && value % test_num != 0)
    //{
    //    test_num--;
    //}

    //int rem = value / test_num;
    //int test_num2 = (int)sqrt(rem);
    //while (test_num2 > 0 && rem % test_num2 != 0)
    //{
    //    test_num2--;
    //}
    //*factor_3 = test_num;        //nz
    //*factor_2 = test_num2;       //ny
    //*factor_1 = rem / test_num2; //nx

    int best_x = 1, best_y = 1, best_z = value;
    int min_diff = value; // start with a large difference
    for (int x = 1; x <= value; ++x) {
        if (value % x != 0) continue;
        int yz = value / x;
        for (int y = 1; y <= yz; ++y) {
            if (yz % y != 0) continue;
            int z = yz / y;
            int arr[3] = {x, y, z};
            std::sort(arr, arr + 3);
            int diff = arr[2] - arr[0];
            if (diff < min_diff) {
                min_diff = diff;
                best_x = arr[0];
                best_y = arr[1];
                best_z = arr[2];
            }
        }
    }
    *factor_1 = best_x;
    *factor_2 = best_y;
    *factor_3 = best_z;
}

// private - exchange boundary information between MPI ranks
void LbmDQ::exchangeBoundaries()
{
//    std::cout << "[Rank " << rank << "] Entering exchangeBoundaries" << std::endl;

    MPI_Barrier(cart_comm);

    // Exchange data for all distribution functions
    for (int d = 0; d < Q; ++d) {
        // North-South exchange
        if (neighbors[NeighborN] != MPI_PROC_NULL || neighbors[NeighborS] != MPI_PROC_NULL) {
    	    MPI_CHECK(MPI_Sendrecv(fPtr[d], 1, faceN, neighbors[NeighborN], TAG_F,
                     fPtr[d], 1, faceS, neighbors[NeighborS], TAG_F,
                     cart_comm, MPI_STATUS_IGNORE));
	}

        // East-West exchange
        if (neighbors[NeighborE] != MPI_PROC_NULL || neighbors[NeighborW] != MPI_PROC_NULL) {
	    MPI_CHECK(MPI_Sendrecv(fPtr[d], 1, faceE, neighbors[NeighborE], TAG_F,
                     fPtr[d], 1, faceW, neighbors[NeighborW], TAG_F,
                     cart_comm, MPI_STATUS_IGNORE));
	}

        // Northeast-Southwest exchange
        if (neighbors[NeighborNE] != MPI_PROC_NULL || neighbors[NeighborSW] != MPI_PROC_NULL) {
	    MPI_CHECK(MPI_Sendrecv(fPtr[d], 1, faceNE, neighbors[NeighborNE], TAG_F,
                     fPtr[d], 1, faceSW, neighbors[NeighborSW], TAG_F,
                     cart_comm, MPI_STATUS_IGNORE));
	}

        // Northwest-Southeast exchange
        if (neighbors[NeighborNW] != MPI_PROC_NULL || neighbors[NeighborSE] != MPI_PROC_NULL) {
	    MPI_CHECK(MPI_Sendrecv(fPtr[d], 1, faceNW, neighbors[NeighborNW], TAG_F,
                     fPtr[d], 1, faceSE, neighbors[NeighborSE], TAG_F,
                     cart_comm, MPI_STATUS_IGNORE));
	}

        // Up-Down exchange
        if (neighbors[NeighborDown] != MPI_PROC_NULL || neighbors[NeighborUp] != MPI_PROC_NULL) {
	    MPI_CHECK(MPI_Sendrecv(fPtr[d], 1, faceZlo, neighbors[NeighborDown], TAG_F,
                     fPtr[d], 1, faceZhi, neighbors[NeighborUp], TAG_F,
                     cart_comm, MPI_STATUS_IGNORE));
	}
    }    

    MPI_Barrier(cart_comm);

    // density
    if (neighbors[NeighborN] != MPI_PROC_NULL || neighbors[NeighborS] != MPI_PROC_NULL) {
        MPI_Sendrecv(density, 1, faceN, neighbors[NeighborN], TAG_D,
                 density, 1, faceS, neighbors[NeighborS], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborE] != MPI_PROC_NULL || neighbors[NeighborW] != MPI_PROC_NULL) {
        MPI_Sendrecv(density, 1, faceE, neighbors[NeighborE], TAG_D,
                 density, 1, faceW, neighbors[NeighborW], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborNE] != MPI_PROC_NULL || neighbors[NeighborSW] != MPI_PROC_NULL) {
        MPI_Sendrecv(density, 1, faceNE, neighbors[NeighborNE], TAG_D,
                 density, 1, faceSW, neighbors[NeighborSW], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborNW] != MPI_PROC_NULL || neighbors[NeighborSE] != MPI_PROC_NULL) {
        MPI_Sendrecv(density, 1, faceNW, neighbors[NeighborNW], TAG_D,
                 density, 1, faceSE, neighbors[NeighborSE], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborDown] != MPI_PROC_NULL || neighbors[NeighborUp] != MPI_PROC_NULL) {
        MPI_Sendrecv(density, 1, faceZlo, neighbors[NeighborDown], TAG_D,
                 density, 1, faceZhi, neighbors[NeighborUp], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    // velocity_x
    if (neighbors[NeighborN] != MPI_PROC_NULL || neighbors[NeighborS] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_x, 1, faceN, neighbors[NeighborN], TAG_VX,
                 velocity_x, 1, faceS, neighbors[NeighborS], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborE] != MPI_PROC_NULL || neighbors[NeighborW] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_x, 1, faceE, neighbors[NeighborE], TAG_VX,
                 velocity_x, 1, faceW, neighbors[NeighborW], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborNE] != MPI_PROC_NULL || neighbors[NeighborSW] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_x, 1, faceNE, neighbors[NeighborNE], TAG_VX,
                 velocity_x, 1, faceSW, neighbors[NeighborSW], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborNW] != MPI_PROC_NULL || neighbors[NeighborSE] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_x, 1, faceNW, neighbors[NeighborNW], TAG_VX,
                 velocity_x, 1, faceSE, neighbors[NeighborSE], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborDown] != MPI_PROC_NULL || neighbors[NeighborUp] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_x, 1, faceZlo, neighbors[NeighborDown], TAG_VX,
                 velocity_x, 1, faceZhi, neighbors[NeighborUp], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    // velocity_y
    if (neighbors[NeighborN] != MPI_PROC_NULL || neighbors[NeighborS] != MPI_PROC_NULL) {
	MPI_Sendrecv(velocity_y, 1, faceN, neighbors[NeighborN], TAG_VY,
                 velocity_y, 1, faceS, neighbors[NeighborS], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborE] != MPI_PROC_NULL || neighbors[NeighborW] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_y, 1, faceE, neighbors[NeighborE], TAG_VY,
                 velocity_y, 1, faceW, neighbors[NeighborW], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborNE] != MPI_PROC_NULL || neighbors[NeighborSW] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_y, 1, faceNE, neighbors[NeighborNE], TAG_VY,
                 velocity_y, 1, faceSW, neighbors[NeighborSW], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborNW] != MPI_PROC_NULL || neighbors[NeighborSE] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_y, 1, faceNW, neighbors[NeighborNW], TAG_VY,
                 velocity_y, 1, faceSE, neighbors[NeighborSE], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborDown] != MPI_PROC_NULL || neighbors[NeighborUp] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_y, 1, faceZlo, neighbors[NeighborDown], TAG_VY,
                 velocity_y, 1, faceZhi, neighbors[NeighborUp], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    // velocity_z
    if (neighbors[NeighborN] != MPI_PROC_NULL || neighbors[NeighborS] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_z, 1, faceN, neighbors[NeighborN], TAG_VZ,
                 velocity_z, 1, faceS, neighbors[NeighborS], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborE] != MPI_PROC_NULL || neighbors[NeighborW] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_z, 1, faceE, neighbors[NeighborE], TAG_VZ,
                 velocity_z, 1, faceW, neighbors[NeighborW], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborNE] != MPI_PROC_NULL || neighbors[NeighborSW] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_z, 1, faceNE, neighbors[NeighborNE], TAG_VZ,
                 velocity_z, 1, faceSW, neighbors[NeighborSW], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborNW] != MPI_PROC_NULL || neighbors[NeighborSE] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_z, 1, faceNW, neighbors[NeighborNW], TAG_VZ,
                 velocity_z, 1, faceSE, neighbors[NeighborSE], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    if (neighbors[NeighborDown] != MPI_PROC_NULL || neighbors[NeighborUp] != MPI_PROC_NULL) {
        MPI_Sendrecv(velocity_z, 1, faceZlo, neighbors[NeighborDown], TAG_VZ,
                 velocity_z, 1, faceZhi, neighbors[NeighborUp], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(cart_comm);

//    std::cout << "[Rank " << rank << "] Exiting exchangeBoundaries" << std::endl;
}

#endif // _LBMDQ_MPI_HPP_
