#ifndef _LBMD3Q15_MPI_HPP_
#define _LBMD3Q15_MPI_HPP_

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <mpi.h>

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

// Lattice-Boltzman Methods CFD simulation
class LbmD3Q15
{
    public:
        enum FluidProperty {None, Density, Speed, Vorticity};

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
	static constexpr int Q = 15;
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
        float *f_0, *f_1, *f_2, *f_3, *f_4, *f_5, *f_6, *f_7, *f_8, *f_9, *f_10, *f_11, *f_12, *f_13, *f_14;
        float *dbl_arrays;
        uint32_t block_width, block_height, block_depth;
	float **fPtr;

	// Helper function to print memory usage
        void printMemoryUsage(const char* label, size_t bytes) {
            double mb = bytes / (1024.0 * 1024.0);
            double gb = mb / 1024.0;
            //if (rank == 0) {
            //    std::cout << "Memory usage - " << label << ": " << mb << " MB (" << gb << " GB)" << std::endl;
            //}
        }

	// Helper functions
        inline int idx3D(int x, int y, int z) const {
            return x + dim_x * (y + dim_y * z);
        }

        inline float& f_at(int d, int x, int y, int z) const {
            return fPtr[d][idx3D(x, y, z)];
        }

        void setEquilibrium(int x, int y, int z, double new_velocity_x, double new_velocity_y, double new_velocity_z, double new_density);
        void getClosestFactors3(int value, int *factor_1, int *factor_2, int *factor_3);

    public:
        LbmD3Q15(uint32_t width, uint32_t height, uint32_t depth, double scale, int task_id, int num_tasks);
        ~LbmD3Q15();

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
};

// constructor
LbmD3Q15::LbmD3Q15(uint32_t width, uint32_t height, uint32_t depth, double scale, int task_id, int num_tasks)
{
    rank = task_id;
    num_ranks = num_tasks;
    speed_scale = scale;
    stored_property = None;

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
    row = rank / n_x;

    //New
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

    // Debugging
    //if (rank == 0) {
    //    std::cout << "Debug - Initial values:" << std::endl;
    //    std::cout << "  width/height/depth: " << width << " " << height << " " << depth << std::endl;
    //    std::cout << "  num_x/y/z: " << num_x << " " << num_y << " " << num_z << std::endl;
    //    std::cout << "  block_width/height/depth: " << block_width << " " << block_height << " " << block_depth << std::endl;
    //    std::cout << "  dim_x/y/z: " << dim_x << " " << dim_y << " " << dim_z << std::endl;
    //}

    // create data types for exchanging data with neighbors
    int sizes3D[3] = {int(dim_z), int(dim_y), int(dim_x)};
    int subsize3D[3] = {int(num_z), int(num_y), int(num_x)};
    int offsets3D[3] = {int(offset_z), int(offset_y), int(offset_x)};

    // More debugging
    //if (rank == 0) {
    //    std::cout << "Debug - sizes3D: " << sizes3D[0] << " " << sizes3D[1] << " " << sizes3D[2] << std::endl;
    //    std::cout << "Debug - subsize3D: " << subsize3D[0] << " " << subsize3D[1] << " " << subsize3D[2] << std::endl;
    //    std::cout << "Debug - offsets3D: " << offsets3D[0] << " " << offsets3D[1] << " " << offsets3D[2] << std::endl;
    //    std::cout << "Debug - dims: " << dim_x << " " << dim_y << " " << dim_z << std::endl;
    //    std::cout << "Debug - num: " << num_x << " " << num_y << " " << num_z << std::endl;
    //    std::cout << "Debug - offset: " << offset_x << " " << offset_y << " " << offset_z << std::endl;
    //}

    MPI_Type_create_subarray(3, sizes3D, subsize3D, offsets3D, MPI_ORDER_C, MPI_DOUBLE, &own_scalar);
    MPI_Type_commit(&own_scalar);
    MPI_Type_create_subarray(3, sizes3D, subsize3D, offsets3D, MPI_ORDER_C, MPI_BYTE, &own_bool);
    MPI_Type_commit(&own_bool);

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
        
        MPI_Type_create_subarray(3, sizes3D, osub, ooffset, MPI_ORDER_C, MPI_DOUBLE, &other_scalar[r]);
        MPI_Type_commit(&other_scalar[r]);
        MPI_Type_create_subarray(3, sizes3D, osub, ooffset, MPI_ORDER_C, MPI_BYTE, &other_bool[r]);
        MPI_Type_commit(&other_bool[r]); 
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

    recv_buf = new float[total_x * total_y * total_z];
    brecv_buf = new bool[total_x * total_y * total_z];
    
    uint32_t size = dim_x * dim_y * dim_z;
    size_t total_memory = 0;

    // allocate all float arrays at once
    dbl_arrays = new float[21 * size];
    total_memory += 21 * size * sizeof(float);
    printMemoryUsage("Main arrays", total_memory);

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

    density    = dbl_arrays + (15*size);
    velocity_x = dbl_arrays + (16*size);
    velocity_y = dbl_arrays + (17*size);
    velocity_z = dbl_arrays + (18*size);
    vorticity  = dbl_arrays + (19*size);
    speed      = dbl_arrays + (20*size);
    
    // allocate boolean array
    barrier = new bool[size];
    total_memory += size * sizeof(bool);
    printMemoryUsage("Barrier array", size * sizeof(bool));

    // allocate receive buffers
    recv_buf = new float[total_x * total_y * total_z];
    brecv_buf = new bool[total_x * total_y * total_z];
    total_memory += (total_x * total_y * total_z) * (sizeof(float) + sizeof(bool));
    printMemoryUsage("Receive buffers", (total_x * total_y * total_z) * (sizeof(float) + sizeof(bool)));

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

    printMemoryUsage("Total", total_memory);
}

// destructor
LbmD3Q15::~LbmD3Q15()
{
    //if (rank == 0) std::cout << "Starting destructor..." << std::endl;

    // Free MPI types
    //if (rank == 0) std::cout << "Freeing MPI types..." << std::endl;
    
    if (faceXlo != MPI_DATATYPE_NULL) MPI_Type_free(&faceXlo);
    if (faceXhi != MPI_DATATYPE_NULL) MPI_Type_free(&faceXhi);
    if (faceYlo != MPI_DATATYPE_NULL) MPI_Type_free(&faceYlo);
    if (faceYhi != MPI_DATATYPE_NULL) MPI_Type_free(&faceYhi);
    if (faceZlo != MPI_DATATYPE_NULL) MPI_Type_free(&faceZlo);
    if (faceZhi != MPI_DATATYPE_NULL) MPI_Type_free(&faceZhi);
    if (faceN != MPI_DATATYPE_NULL) MPI_Type_free(&faceN);
    if (faceS != MPI_DATATYPE_NULL) MPI_Type_free(&faceS);
    if (faceE != MPI_DATATYPE_NULL) MPI_Type_free(&faceE);
    if (faceW != MPI_DATATYPE_NULL) MPI_Type_free(&faceW);
    if (faceNE != MPI_DATATYPE_NULL) MPI_Type_free(&faceNE);
    if (faceNW != MPI_DATATYPE_NULL) MPI_Type_free(&faceNW);
    if (faceSE != MPI_DATATYPE_NULL) MPI_Type_free(&faceSE);
    if (faceSW != MPI_DATATYPE_NULL) MPI_Type_free(&faceSW);
    if (own_scalar != MPI_DATATYPE_NULL) MPI_Type_free(&own_scalar);
    if (own_bool != MPI_DATATYPE_NULL) MPI_Type_free(&own_bool);

    // Free other MPI types
    //if (rank == 0) std::cout << "Freeing other MPI types..." << std::endl;
    if (other_scalar != nullptr) {
        for (int i=0; i < num_ranks; i++) {
            if (other_scalar[i] != MPI_DATATYPE_NULL) {
                MPI_Type_free(&other_scalar[i]);
            }
        }
        delete[] other_scalar;
    }

    if (other_bool != nullptr) {
        for (int i=0; i < num_ranks; i++) {
            if (other_bool[i] != MPI_DATATYPE_NULL) {
                MPI_Type_free(&other_bool[i]);
            }
        }
        delete[] other_bool;
    }

    // Free arrays
    //if (rank == 0) std::cout << "Freeing arrays..." << std::endl;
    if (rank_local_size != nullptr) {
        //if (rank == 0) std::cout << "Freeing rank_local_size..." << std::endl;
        delete[] rank_local_size;
    }
    if (rank_local_start != nullptr) {
        //if (rank == 0) std::cout << "Freeing rank_local_start..." << std::endl;
        delete[] rank_local_start;
    }
    if (barrier != nullptr) {
        //if (rank == 0) std::cout << "Freeing barrier..." << std::endl;
        delete[] barrier;
    }
    if (recv_buf != nullptr) {
        //if (rank == 0) std::cout << "Freeing recv_buf..." << std::endl;
        delete[] recv_buf;
    }
    if (brecv_buf != nullptr) {
        //if (rank == 0) std::cout << "Freeing brecv_buf..." << std::endl;
        delete[] brecv_buf;
    }

    // Free fPtr before dbl_arrays since fPtr points into dbl_arrays
    if (fPtr != nullptr) {
        //if (rank == 0) std::cout << "Freeing fPtr..." << std::endl;
        delete[] fPtr;
    }
    
    // Finally free the main array that contains all the float data
    if (dbl_arrays != nullptr) {
        //if (rank == 0) std::cout << "Freeing dbl_arrays..." << std::endl;
        delete[] dbl_arrays;
    }

    //if (rank == 0) std::cout << "Destructor completed successfully." << std::endl;
}

// initialize barrier based on selected type
void LbmD3Q15::initBarrier(std::vector<Barrier*> barriers)
{
	// clear barrier to all `false`
	memset(barrier, 0, dim_x * dim_y * dim_z);

	// set barrier to `true` where horizontal or vertical barriers exist
	int sx = (offset_x == 0) ? 0 : offset_x - 1;
	int sy = (offset_y == 0) ? 0 : offset_y - 1;
	int sz = (offset_z == 0) ? 0 : offset_z - 1;
	int i, j;
        for (i = 0; i < barriers.size(); i++) {
          if (barriers[i]->getType() == Barrier::Type::HORIZONTAL) {
              int y = barriers[i]->getY1() - sy;
              if (y >= 0 && y < dim_y) {
                  for (j = barriers[i]->getX1(); j <= barriers[i]->getX2(); j++) {
                      int x = j - sx;
                    if (x >= 0 && x < dim_x) {
			for (int k = sz; k < dim_z - sz; ++k) {
			    barrier[idx3D(x,y,k)] = true;
			}
		    }
		  }
	      }
	  }
	  else {                // Barrier::VERTICAL
              int x = barriers[i]->getX1() - sx;
              if (x >= 0 && x < dim_x) {
                  for (j = barriers[i]->getY1(); j <= barriers[i]->getY2(); j++) {
                      int y = j - sy;
                      if (y >= 0 && y < dim_y) {
		    	  // extrude vertical line through every Z-layer
			  for (int k = sz; k < dim_z - sz; ++k) {
		    	      barrier[idx3D(x,y,k)] = true;
			  }
		      }
		  }
	      }
	  }
	}
}
	
// initialize fluid
void LbmD3Q15::initFluid(double physical_speed)
{
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
}

void LbmD3Q15::updateFluid(double physical_speed)
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
void LbmD3Q15::collide(double viscosity)
{
	//if (rank == 0) std::cout << "Starting collide (viscosity=" << viscosity << ")" << std::endl;
	int i, j, row, idx;
	double omega = 1.0 / (3.0 * viscosity + 0.5); //reciprocal of relaxation time
	
	for (j = 1; j < dim_y -1; j++)
	{
		row = j * dim_x;
		for (i = 1; i < dim_x - 1; ++i)
		{
			idx = row + i;

			double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
			for (int d = 0; d < 15; ++d)
			{
				double fv = fPtr[d][idx];
				rho += fv;
				ux  += fv * cD3Q15[d][0];
				uy  += fv * cD3Q15[d][1];
				uz  += fv * cD3Q15[d][2];
			}
			density[idx] = rho;
			ux /= rho; uy /= rho; uz /= rho;
			velocity_x[idx] = ux;
			velocity_y[idx] = uy;
			velocity_z[idx] = uz;

			double usqr = ux*ux + uy*uy + uz*uz;
			for (int d = 0; d < 15; ++d)
			{
				double cu = 3.0 * (cD3Q15[d][0]*ux + cD3Q15[d][1]*uy + cD3Q15[d][2]*uz);
				double feq = wD3Q15[d] * rho * (1.0 + cu + 0.5*cu*cu - 1.5*usqr);
				fPtr[d][idx] += omega * (feq - fPtr[d][idx]);
			}
		}
	}
        //if (rank == 0) std::cout << "Completed collide" << std::endl;
}
	
// particle streaming
void LbmD3Q15::stream()
{
	//if (rank == 0) std::cout << "Starting stream" << std::endl;
	
	size_t slice = static_cast<size_t>(dim_x) * dim_y * dim_z;
	float* f_Old = new float[Q * slice];
	std::memcpy(f_Old, f, Q * slice * sizeof(float));

	//if (rank == 0) std::cout << "  Streaming distributions" << std::endl;
	for (int k = start_z; k < dim_z - start_z; ++k) {
		for (int j = start_y; j < dim_y - start_y; ++j) {
			for (int i = start_x; i < dim_x - start_x; ++i) {
				int idx = idx3D(i, j, k);
				for (int d = 0; d < 15; ++d) {
					int ni = i + cD3Q15[d][0];
					int nj = j + cD3Q15[d][1];
					int nk = k + cD3Q15[d][2];
					int nidx = idx3D(ni, nj, nk);

					f_at(d, ni, nj, nk) = f_Old[d * slice + idx];
				}
			}
		}
	}

	delete[] f_Old;
	//if (rank == 0) std::cout << "Completed stream" << std::endl;
}

// particle streaming bouncing back off of barriers
void LbmD3Q15::bounceBackStream()
{
	//if (rank == 0) std::cout << "Starting bounceBackStream" << std::endl;
	size_t slice = static_cast<size_t>(dim_x) * dim_y * dim_z;
	float* f_Old = new float[Q * slice];
	std::memcpy(f_Old, f, Q * slice * sizeof(float));

	//if (rank == 0) std::cout << "  Streaming with bounce-back" << std::endl;
	for (int k = start_z; k < dim_z - start_z; ++k)
	{
		for (int j = start_y; j < dim_y - start_y; ++j)
		{
			for (int i = start_x; i < dim_x - start_x; ++i)
			{
				int idx = idx3D(i, j, k);

				for (int d = 1; d < Q; ++d)
				{
					int ni = i + cD3Q15[d][0];
					int nj = j + cD3Q15[d][1];
					int nk = k + cD3Q15[d][2];
					int nidx = idx3D(ni, nj, nk);

					if (barrier[nidx])
					{
						int od = 0;
						for (int dd = 1; dd < Q; ++dd)
						{
							if (cD3Q15[dd][0] == -cD3Q15[d][0] && cD3Q15[dd][1] == -cD3Q15[d][1] && cD3Q15[dd][2] == -cD3Q15[d][2])
							{
								od = dd;
								break;
							}
						}
						fPtr[d][idx] = f_Old[od * slice + idx];
					}
				}
			}
		}
	}

	delete[] f_Old;
	//if (rank == 0) std::cout << "Completed bounceBackStream" << std::endl;	
}
	
// check if simulation has become unstable (if so, more time steps are required)
bool LbmD3Q15::checkStability()
{
    int i, k, idx;
    bool stable = true;
    int j = dim_y / 2;
    for (k = 0; k < dim_z; k++)
    {
	for (i = 0; i < dim_x; i++)
	    {
	        idx = idx3D(i, j, k);
		if (density[idx] <= 0)
		{
		    stable = false;
		}
	    }
    }
    return stable;
}

// compute speed (magnitude of velocity vector)
void LbmD3Q15::computeSpeed()
{
    int i, j, k, idx;
    for (k = 1; k < dim_z - 1; k++)
    {
	for (j = 1; j < dim_y - 1; j++)
	{
            for (i = 1; i < dim_x - 1; i++)
            {
		idx = idx3D(i, j, k);
		speed[idx] = sqrt(velocity_x[idx] * velocity_x[idx] + velocity_y[idx] * velocity_y[idx] + velocity_z[idx] * velocity_z[idx]);
	    }
	}
    }
}

// compute vorticity (rotational velocity)
void LbmD3Q15::computeVorticity()
{
    int i; int j; int k; int idx;

    for (k = 1; k < dim_z - 1; k++)
    {
	for (j = 1; j < dim_y -1; j++)
	{
	    for (i = 1; i < dim_x - 1; i++)
	    {
		idx = idx3D(i, j, k);

		double wx = (velocity_z[idx3D(i, j + 1, k)] - velocity_z[idx3D(i, j - 1, k)]) - (velocity_y[idx3D(i, j, k + 1)] - velocity_y[idx3D(i, j, k - 1)]);

		double wy = (velocity_z[idx3D(i, j, k + 1)] - velocity_z[idx3D(i, j, k - 1)]) - (velocity_y[idx3D(i + 1, j, k)] - velocity_y[idx3D(i - 1, j, k)]);

		double wz = (velocity_z[idx3D(i + 1, j, k)] - velocity_z[idx3D(i - 1, j, k)]) - (velocity_y[idx3D(i, j + 1, k)] - velocity_y[idx3D(i, j - 1, k)]);

		vorticity[idx] = sqrt(wx*wx + wy*wy + wz*wz);
	    }
	}
    }
}

// gather all data on rank 0
void LbmD3Q15::gatherDataOnRank0(FluidProperty property)
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
uint32_t LbmD3Q15::getDimX()
{
    return dim_x;
}

// get width of sub-area this task owns (including ghost cells)
uint32_t LbmD3Q15::getDimY()
{
    return dim_y;
}

// get width of sub-area this task owns (including ghost cells)
uint32_t LbmD3Q15::getDimZ()
{
    return dim_z;
}

// get width of total area of simulation
uint32_t LbmD3Q15::getTotalDimX()
{
    return total_x;
}

// get width of total area of simulation
uint32_t LbmD3Q15::getTotalDimY()
{
    return total_y;
}

// get width of total area of simulation
uint32_t LbmD3Q15::getTotalDimZ()
{
    return total_z;
}

// get x offset into overall domain where this sub-area esxists
uint32_t LbmD3Q15::getOffsetX()
{
    return offset_x;
}

// get y offset into overall domain where this sub-area esxists
uint32_t LbmD3Q15::getOffsetY()
{
    return offset_y;
}

// get z offset into overall domain where this sub-area esxists
uint32_t LbmD3Q15::getOffsetZ()
{
    return offset_z;
}

// get x start for valid data (0 if no ghost cell on left, 1 if there is a ghost cell on left)
uint32_t LbmD3Q15::getStartX()
{
    return start_x;
}

// get y start for valid data (0 if no ghost cell on top, 1 if there is a ghost cell on top)
uint32_t LbmD3Q15::getStartY()
{
    return start_y;
}

// get z start for valid data (0 if no ghost cell on top, 1 if there is a ghost cell on top)
uint32_t LbmD3Q15::getStartZ()
{
    return start_z;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmD3Q15::getSizeX()
{
    return num_x;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmD3Q15::getSizeY()
{
    return num_y;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmD3Q15::getSizeZ()
{
    return num_z;
}

// get the local width and height of a particular rank's data
uint32_t* LbmD3Q15::getRankLocalSize(int rank)
{
    return rank_local_size + (2 * rank);
}

// get the local x and y start of a particular rank's data
uint32_t* LbmD3Q15::getRankLocalStart(int rank)
{
    return rank_local_start + (2 * rank);
}

// get barrier array
bool* LbmD3Q15::getBarrier()
{
    if (rank != 0) return NULL;
    return brecv_buf;
}

// private - set fluid equalibrium
void LbmD3Q15::setEquilibrium(int x, int y, int z, double new_velocity_x, double new_velocity_y, double new_velocity_z, double new_density)
{
	int idx = idx3D(x, y, z);

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
		double cu = 3.0 * (cD3Q15[d][0] * ux + cD3Q15[d][1] * uy + cD3Q15[d][2] * uz);
		f_at(d, x, y, z) = wD3Q15[d] * new_density * (1.0 + cu + 0.5*cu*cu - 1.5*usq);
	}
}

// private - get 3 factors of a given number that are closest to each other
void LbmD3Q15::getClosestFactors3(int value, int *factor_1, int *factor_2, int *factor_3)
{
    int test_num = (int)cbrt(value);
    while (test_num > 0 && value % test_num != 0)
    {
	test_num--;
    }

    int rem = value / test_num;
    int test_num2 = (int)sqrt(rem);
    while (test_num2 > 0 && rem % test_num2 != 0)
    {
        test_num2--;
    }
    *factor_3 = test_num;        //nz
    *factor_2 = test_num2;       //ny
    *factor_1 = rem / test_num2; //nx
}

// private - exchange boundary information between MPI ranks
void LbmD3Q15::exchangeBoundaries()
{
    //if (rank == 0) std::cout << "Starting exchangeBoundaries" << std::endl;

    // Exchange data for all distribution functions
    for (int d = 0; d < Q; ++d) {
	//if (rank == 0 && d % 5 == 0) std::cout << "  Exchanging distribution " << d << std::endl;
        // North-South exchange
        MPI_Sendrecv(fPtr[d], 1, faceN, neighbors[NeighborN], TAG_F,
                     fPtr[d], 1, faceS, neighbors[NeighborS], TAG_F,
                     cart_comm, MPI_STATUS_IGNORE);

        // East-West exchange
        MPI_Sendrecv(fPtr[d], 1, faceE, neighbors[NeighborE], TAG_F,
                     fPtr[d], 1, faceW, neighbors[NeighborW], TAG_F,
                     cart_comm, MPI_STATUS_IGNORE);

        // Northeast-Southwest exchange
        MPI_Sendrecv(fPtr[d], 1, faceNE, neighbors[NeighborNE], TAG_F,
                     fPtr[d], 1, faceSW, neighbors[NeighborSW], TAG_F,
                     cart_comm, MPI_STATUS_IGNORE);

        // Northwest-Southeast exchange
        MPI_Sendrecv(fPtr[d], 1, faceNW, neighbors[NeighborNW], TAG_F,
                     fPtr[d], 1, faceSE, neighbors[NeighborSE], TAG_F,
                     cart_comm, MPI_STATUS_IGNORE);

        // Up-Down exchange
        MPI_Sendrecv(fPtr[d], 1, faceZlo, neighbors[NeighborDown], TAG_F,
                     fPtr[d], 1, faceZhi, neighbors[NeighborUp], TAG_F,
                     cart_comm, MPI_STATUS_IGNORE);
    }

    //if (rank == 0) std::cout << "  Exchanging density field" << std::endl;
    // density
    MPI_Sendrecv(density, 1, faceN, neighbors[NeighborN], TAG_D,
                 density, 1, faceS, neighbors[NeighborS], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(density, 1, faceE, neighbors[NeighborE], TAG_D,
                 density, 1, faceW, neighbors[NeighborW], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(density, 1, faceNE, neighbors[NeighborNE], TAG_D,
                 density, 1, faceSW, neighbors[NeighborSW], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(density, 1, faceNW, neighbors[NeighborNW], TAG_D,
                 density, 1, faceSE, neighbors[NeighborSE], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(density, 1, faceZlo, neighbors[NeighborDown], TAG_D,
                 density, 1, faceZhi, neighbors[NeighborUp], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);

    //if (rank == 0) std::cout << "  Exchanging velocity_x field" << std::endl;
    // velocity_x
    MPI_Sendrecv(velocity_x, 1, faceN, neighbors[NeighborN], TAG_VX,
                 velocity_x, 1, faceS, neighbors[NeighborS], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_x, 1, faceE, neighbors[NeighborE], TAG_VX,
                 velocity_x, 1, faceW, neighbors[NeighborW], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_x, 1, faceNE, neighbors[NeighborNE], TAG_VX,
                 velocity_x, 1, faceSW, neighbors[NeighborSW], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_x, 1, faceNW, neighbors[NeighborNW], TAG_VX,
                 velocity_x, 1, faceSE, neighbors[NeighborSE], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_x, 1, faceZlo, neighbors[NeighborDown], TAG_VX,
                 velocity_x, 1, faceZhi, neighbors[NeighborUp], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);

    //if (rank == 0) std::cout << "  Exchanging velocity_y field" << std::endl;
    // velocity_y
    MPI_Sendrecv(velocity_y, 1, faceN, neighbors[NeighborN], TAG_VY,
                 velocity_y, 1, faceS, neighbors[NeighborS], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_y, 1, faceE, neighbors[NeighborE], TAG_VY,
                 velocity_y, 1, faceW, neighbors[NeighborW], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_y, 1, faceNE, neighbors[NeighborNE], TAG_VY,
                 velocity_y, 1, faceSW, neighbors[NeighborSW], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_y, 1, faceNW, neighbors[NeighborNW], TAG_VY,
                 velocity_y, 1, faceSE, neighbors[NeighborSE], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_y, 1, faceZlo, neighbors[NeighborDown], TAG_VY,
                 velocity_y, 1, faceZhi, neighbors[NeighborUp], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);

    //if (rank == 0) std::cout << "  Exchanging velocity_z field" << std::endl;
    // velocity_z
    MPI_Sendrecv(velocity_z, 1, faceN, neighbors[NeighborN], TAG_VZ,
                 velocity_z, 1, faceS, neighbors[NeighborS], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_z, 1, faceE, neighbors[NeighborE], TAG_VZ,
                 velocity_z, 1, faceW, neighbors[NeighborW], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_z, 1, faceNE, neighbors[NeighborNE], TAG_VZ,
                 velocity_z, 1, faceSW, neighbors[NeighborSW], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_z, 1, faceNW, neighbors[NeighborNW], TAG_VZ,
                 velocity_z, 1, faceSE, neighbors[NeighborSE], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_z, 1, faceZlo, neighbors[NeighborDown], TAG_VZ,
                 velocity_z, 1, faceZhi, neighbors[NeighborUp], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);
    //if (rank == 0) std::cout << "Completed exchangeBoundaries" << std::endl;
}

#endif // _LBMD3Q15_MPI_HPP_
