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
        //uint32_t *rank_local_size;
        //uint32_t *rank_local_start;
        double speed_scale;
	static constexpr int Q = 15;
        double *f;
        double *density;
        double *velocity_x;
        double *velocity_y;
	double *velocity_z;
        double *vorticity;
        double *speed;
        bool *barrier;
        FluidProperty stored_property;
        double *recv_buf;
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

    private:
        // Add missing member variables
        double *f_0, *f_1, *f_2, *f_3, *f_4, *f_5, *f_6, *f_7, *f_8, *f_9, *f_10, *f_11, *f_12, *f_13, *f_14;
        double *dbl_arrays;
        uint32_t block_width, block_height, block_depth;
        int idx3D(int x, int y, int z) const;
        double& f_at(int d, int x, int y, int z) const;

        void setEquilibrium(int x, int y, int z, double new_velocity_x, double new_velocity_y, double new_velocity_z, double new_density);
        void getClosestFactors3(int value, int *factor_1, int *factor_2, int *factor_3);
        void exchangeBoundaries();

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
        //uint32_t* getRankLocalSize(int rank);
        //uint32_t* getRankLocalStart(int rank);
        bool* getBarrier();
        double* getDensity();
        double* getVelocityX();
        double* getVelocityY();
	double* getVelocityZ();
        double* getVorticity();
        double* getSpeed();
	static constexpr int TAG_F  = 100;
        static constexpr int TAG_D  = 101;
        static constexpr int TAG_VX = 102;
        static constexpr int TAG_VY = 103;
        static constexpr int TAG_VZ = 104;
	static constexpr int TAB_B  = 105;
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
    int neighbor_cols, neighbor_rows;
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
    neighbor_cols = (num_ranks == 1) ? 0 : ((col == 0 || col == n_x-1) ? 1 : 2);
    neighbor_rows = (num_ranks == 1) ? 0 : ((row == 0 || row == n_y-1) ? 1 : 2);
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

    // create data types for exchanging data with neighbors
    int sizes3D[3] = {int(dim_z), int(dim_y), int(dim_x)};
    int subsize3D[3] = {int(num_z), int(num_y), int(num_x)};
    int offsets3D[3] = {int(offset_z), int(offset_y), int(offset_x)};

    MPI_Type_create_subarray(3, sizes3D, subsize3D, offsets3D, MPI_ORDER_C, MPI_DOUBLE, &own_scalar);
    MPI_Type_commit(&own_scalar);
    MPI_Type_create_subarray(3, sizes3D, subsize3D, offsets3D, MPI_ORDER_C, MPI_BYTE, &own_bool);
    MPI_Type_commit(&own_bool);

    other_scalar = new MPI_Datatype[num_ranks];
    other_bool = new MPI_Datatype[num_ranks];
    for (int r = 0; r < num_ranks; r++)
    {
	int col = r % n_x;
	int row = (r / n_x) % n_y;
	int layer = r / (n_x * n_y);

	    int osub[3] = {
		int(chunk_d + ((layer < extra_d) ? 1 : 0)),
		int(chunk_h + ((row < extra_h) ? 1 : 0)),
		int(chunk_w + ((width < extra_w) ? 1 : 0)),
	    };
	    
	    int ooffset[3] = {
		int(layer * chunk_d + std::min(layer, extra_d)),
		int(row   * chunk_h + std::min(row,   extra_h)),
		int(col   * chunk_w + std::min(col,   extra_w))
	    }
	    
	    MPI_Type_create_subarray(3, sizes3D, osub, ooffset, MPI_ORDER_C, MPI_DOUBLE, &other_scalar[r]);
    	    MPI_Type_commit(&other_scalar[r]);
    	    MPI_Type_create_subarray(3, sizes3D, osub, ooffset, MPI_ORDER_C, MPI_BYTE,   &other_bool[r]);
    	    MPI_Type_commit(&other_bool[r]);
    }

    // create data types for exchanging data with neighbors
    //int block_width, block_height, array[2], subsize[2], offsets[2];
    //block_width = num_x + neighbor_cols;
    //block_height = num_y + neighbor_rows;
    //array[0] = block_height;
    //array[1] = block_width;
    //subsize[0] = num_y;
    //subsize[1] = 1;
    //offsets[0] = start_y;
    //offsets[1] = 0;
    //MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[LeftBoundaryCol]);
    //MPI_Type_commit(&columns_2d[LeftBoundaryCol]); // left boundary column
    //offsets[1] = start_x;
    //MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[LeftCol]);
    //MPI_Type_commit(&columns_2d[LeftCol]); // left column
    //offsets[1] = start_x + num_x - 1;
    //MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[RightCol]);
    //MPI_Type_commit(&columns_2d[RightCol]); // right column
    //offsets[1] = block_width - 1;
    //MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[RightBoundaryCol]);
    //MPI_Type_commit(&columns_2d[RightBoundaryCol]); // right boundary column
    //// create data types for gathering data 
    //subsize[1] = num_x;
    //offsets[1] = start_x;
    //MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &own_scalar);
    //MPI_Type_commit(&own_scalar);
    //MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_BYTE, &own_bool);
    //MPI_Type_commit(&own_bool);
    //other_scalar = new MPI_Datatype[num_ranks];
    //other_bool = new MPI_Datatype[num_ranks];
   
    //int i;
    //int other_col;
    //int other_row;
    //array[0] = height;
    //array[1] = width;
    ////rank_local_size = new uint32_t[2 * num_ranks];
    ////rank_local_start = new uint32_t[2 * num_ranks];
    //for (i=0; i<num_ranks; i++)
    //{
    //    other_col = i % n_x;
    //    other_row = i / n_x;
    //    subsize[0] = chunk_h + ((other_row < extra_h) ? 1 : 0);
    //    subsize[1] = chunk_w + ((other_col < extra_w) ? 1 : 0);
    //    offsets[0] = other_row * chunk_h + std::min(other_row, extra_h);
    //    offsets[1] = other_col * chunk_w + std::min(other_col, extra_w);
    //    //rank_local_size[2 * i + 0] = subsize[1];
    //    //rank_local_size[2 * i + 1] = subsize[0];
    //    //rank_local_start[2 * i + 0] = (other_col == 0) ? 0 : 1;
    //    //rank_local_start[2 * i + 1] = (other_row == 0) ? 0 : 1;
    //    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &other_scalar[i]);
    //    MPI_Type_commit(&other_scalar[i]);
    //    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_BYTE, &other_bool[i]);
    //    MPI_Type_commit(&other_bool[i]);
    //}

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

    inline int idx3D(int x, int y, int z) const {
	    return (z * dim_y + y) * dim_x + x;
    }

    inline double& f_at(int d, int x, int y, int z) const {
	    size_t slice = static_cast<size_t>(dim_x) * dim_y * dim_z;
	    return f[d * slice + idx3D(x,y,z)];
    }

    // set up sub grid for simulation
    total_x = width;
    total_y = height;
    total_z = depth;
    dim_x = block_width;
    dim_y = block_height;
    dim_z = block_depth;
    
    recv_buf = new double[total_x * total_y * total_z];
    brecv_buf = new bool[total_x * total_y * total_z];
    
    uint32_t size = dim_x * dim_y * dim_z;

    // allocate all double arrays at once
    double *dbl_arrays = new double[20 * size];

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

    std::array<double*,15> fPtr = {{
	    f_0, f_1,f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11, f_12, f_13, f_14
    }};

    density    = dbl_arrays + (15*size);
    velocity_x = dbl_arrays + (16*size);
    velocity_y = dbl_arrays + (17*size);
    velocity_z = dbl_arrays + (18*size);
    vorticity  = dbl_arrays + (19*size);
    speed      = dbl_arrays + (20*size);
    
    // allocate boolean array
    barrier = new bool[size];
}

// destructor
LbmD3Q15::~LbmD3Q15()
{
    MPI_Type_free(&faceXlo);
    MPI_Type_free(&faceXhi);
    MPI_Type_free(&faceYlo);
    MPI_Type_free(&faceYhi);
    MPI_Type_free(&faceZlo);
    MPI_Type_free(&faceZhi);

    MPI_Type_free(&faceN);
    MPI_Type_free(&faceS);
    MPI_Type_free(&faceE);
    MPI_Type_free(&faceW);
    MPI_Type_free(&faceNE);
    MPI_Type_free(&faceNW);
    MPI_Type_free(&faceSE);
    MPI_Type_free(&faceSW);

    MPI_Type_free(&own_scalar);
    MPI_Type_free(&own_bool);
    int i;
    for (i=0; i<num_ranks; i++)
    {
        MPI_Type_free(&other_scalar[i]);
        MPI_Type_free(&other_bool[i]);
    }

    delete[] other_scalar;
    delete[] other_bool;
    //delete[] rank_local_size;
    //delete[] rank_local_start;
    delete[] f;
    delete{} density;
    delete[] velocity_x;
    delete[] velocity_y;
    delete[] velocity_z;
    delete[] vorticity;
    delete[] speed;
    delete[] barrier;
    delete[] recv_buf;
    delete[] brecv_buf;
}

// destructor
//LbmD3Q15::~LbmD3Q15()
//{
//    MPI_Type_free(&columns_2d[LeftBoundaryCol]);
//    MPI_Type_free(&columns_2d[LeftCol]);
//    MPI_Type_free(&columns_2d[RightCol]);
//    MPI_Type_free(&columns_2d[RightBoundaryCol]);
//
//    MPI_Type_free(&own_scalar);
//    MPI_Type_free(&own_bool);
//    int i;
//    for (i=0; i<num_ranks; i++)
//    {
//        MPI_Type_free(&other_scalar[i]);
//        MPI_Type_free(&other_bool[i]);
//    }
//
//    //delete[] rank_local_size;
//    //delete[] rank_local_start;
//    delete[] f;
//    delete[] barrier;
//}

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

	
//{
//    
//    // clear barrier to all `false`
//    memset(barrier, 0, dim_x * dim_y);
//    
//    // set barrier to `true` where horizontal or vertical barriers exist
//    int sx = (offset_x == 0) ? 0 : offset_x - 1;
//    int sy = (offset_y == 0) ? 0 : offset_y - 1;
//    int i, j;
//    for (i = 0; i < barriers.size(); i++) {
//        if (barriers[i]->getType() == Barrier::Type::HORIZONTAL) {
//            int y = barriers[i]->getY1() - sy;
//            if (y >= 0 && y < dim_y) {
//                for (j = barriers[i]->getX1(); j <= barriers[i]->getX2(); j++) {
//                    int x = j - sx;
//                    if (x >= 0 && x < dim_x) {
//                        barrier[y * dim_x + x] = true;
//                    }
//                }
//            }
//        }
//        else {                // Barrier::VERTICAL
//            int x = barriers[i]->getX1() - sx;
//            if (x >= 0 && x < dim_x) {
//                for (j = barriers[i]->getY1(); j <= barriers[i]->getY2(); j++) {
//                    int y = j - sy;
//                    if (y >= 0 && y < dim_y) {
//                        barrier[y * dim_x + x] = true;
//                    }
//                }
//            }
//        }
//    }
//}

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

// initialize fluid
//void LbmD3Q15::initFluid(double physical_speed)
//{
//    int i, j, row;
//    double speed = speed_scale * physical_speed;
//    for (j = 0; j < dim_y; j++)
//    {
//        row = j * dim_x;
//        for (i = 0; i < dim_x; i++)
//        {
//            setEquilibrium(i, j, speed, 0.0, 1.0);
//            vorticity[row + i] = 0.0;
//        }
//    }
//}

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

//void LbmD3Q15::updateFluid(double physical_speed)
//{
//    int i;
//    double speed = speed_scale * physical_speed;
//    for (i = 0; i < dim_x; i++)
//    {
//        setEquilibrium(i, 0, speed, 0.0, 1.0);
//        setEquilibrium(i, dim_y - 1, speed, 0.0, 1.0);
//    }
//    for (i = 1; i < dim_y - 1; i++)
//    {
//        setEquilibrium(0, i, speed, 0.0, 1.0);
//        setEquilibrium(dim_x - 1, i, speed, 0.0, 1.0);
//    }
//}

// particle collision
void LbmD3Q15::collide(double viscosity)
{
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

	exchangeBoundaries();
}
	
//{
//    int i, j, row, idx;
//    double omega = 1.0 / (3.0 * viscosity + 0.5); // reciprocal of relaxation time
//    for (j = 1; j < dim_y - 1; j++)
//    {
//        row = j * dim_x;
//        for (i = 1; i < dim_x - 1; i++)
//        {
//            idx = row + i;
//            density[idx] = f_0[idx] + f_1[idx] + f_3[idx] + f_2[idx] + f_4[idx] + f_6[idx] + f_5[idx] + f_8[idx] + f_7[idx];
//            velocity_x[idx] = (f_2[idx] + f_5[idx] + f_7[idx] - f_4[idx] - f_6[idx] - f_8[idx]) / density[idx];
//            velocity_y[idx] = (f_1[idx] + f_5[idx] + f_6[idx] - f_3[idx] - f_7[idx] - f_8[idx]) / density[idx];
//            double one_ninth_density       = (1.0 /  9.0) * density[idx];
//            double four_ninths_density     = (4.0 /  9.0) * density[idx];
//            double one_thirtysixth_density = (1.0 / 36.0) * density[idx];
//            double velocity_3x   = 3.0 * velocity_x[idx];
//            double velocity_3y   = 3.0 * velocity_y[idx];
//            double velocity_x2   = velocity_x[idx] * velocity_x[idx];
//            double velocity_y2   = velocity_y[idx] * velocity_y[idx];
//            double velocity_2xy  = 2.0 * velocity_x[idx] * velocity_y[idx];
//            double vecocity_2    = velocity_x2 + velocity_y2;
//            double vecocity_2_15 = 1.5 * vecocity_2;
//            f_0[idx]  += omega * (four_ninths_density     * (1                                                                 - vecocity_2_15) - f_0[idx]);
//            f_2[idx]  += omega * (one_ninth_density       * (1 + velocity_3x               + 4.5 * velocity_x2                 - vecocity_2_15) - f_2[idx]);
//            f_4[idx]  += omega * (one_ninth_density       * (1 - velocity_3x               + 4.5 * velocity_x2                 - vecocity_2_15) - f_4[idx]);
//            f_1[idx]  += omega * (one_ninth_density       * (1 + velocity_3y               + 4.5 * velocity_y2                 - vecocity_2_15) - f_1[idx]);
//            f_3[idx]  += omega * (one_ninth_density       * (1 - velocity_3y               + 4.5 * velocity_y2                 - vecocity_2_15) - f_3[idx]);
//            f_5[idx] += omega * (one_thirtysixth_density * (1 + velocity_3x + velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15) - f_5[idx]);
//            f_7[idx] += omega * (one_thirtysixth_density * (1 + velocity_3x - velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15) - f_7[idx]);
//            f_6[idx] += omega * (one_thirtysixth_density * (1 - velocity_3x + velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15) - f_6[idx]);
//            f_8[idx] += omega * (one_thirtysixth_density * (1 - velocity_3x - velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15) - f_8[idx]);
//        }
//    }
//
//    exchangeBoundaries();
//}

// particle streaming
void LbmD3Q15::stream()
{
	size_t slice = static_cast<size_t>(dim_x) * dim_y * dim_z;
	double* f_Old = new double[Q * slice];
	std::memcpy(f_Old, f, Q * slice * sizeof(double));

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
	exchangeBoundaries();
	
//{
//    int i, j, row, rowp, rown, idx;
//    for (j = dim_y - 2; j > 0; j--) // first start in NW corner...
//    {
//        row = j * dim_x;
//        rowp = (j - 1) * dim_x;
//        for (i = 1; i < dim_x - 1; i++)
//        {
//            f_1[row + i] =  f_1[rowp + i];
//            f_6[row + i] = f_6[rowp + i + 1];
//        }
//    }
//    for (j = dim_y - 2; j > 0; j--) // then start in NE corner...
//    {
//        row = j * dim_x;
//        rowp = (j - 1) * dim_x;
//        for (i = dim_x - 2; i > 0; i--)
//        {
//            f_2[row + i] =  f_2[row + i - 1];
//            f_5[row + i] = f_5[rowp + i - 1];
//        }
//    }
//    for (j = 1; j < dim_y - 1; j++) // then start in SE corner...
//    {
//        row = j * dim_x;
//        rown = (j + 1) * dim_x;
//        for (i = dim_x - 2; i > 0; i--)
//        {
//            f_3[row + i] =  f_3[rown + i];
//            f_7[row + i] = f_7[rown + i - 1];
//        }
//    }
//    for (j = 1; j < dim_y - 1; j++) // then start in the SW corner...
//    {
//        row = j * dim_x;
//        rown = (j + 1) * dim_x;
//        for (i = 1; i < dim_x - 1; i++)
//        {
//            f_4[row + i] =  f_4[row + i + 1];
//            f_8[row + i] = f_8[rown + i + 1];
//        }
//    }
//
//    exchangeBoundaries();
//}

// particle streaming bouncing back off of barriers
void LbmD3Q15::bounceBackStream()
{
	size_t slice = static_cast<size_t>(dim_x) * dim_y * dim_z;
	double f_Old = new double[Q * slice];
	std::memcpy(f_Old, f, Q * slice * sizeof(double));

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
						f_at(d, i, j, k) = f_Old[od * slice + nidx];
					}
				}
			}
		}
	}

	delete[] f_Old;
}
	
//{
//    int i, j, row, rowp, rown, idx;
//    for (j = 1; j < dim_y - 1; j++) // handle bounce-back from barriers
//    {
//        row = j * dim_x;
//        rowp = (j - 1) * dim_x;
//        rown = (j + 1) * dim_x;
//        for (i = 1; i < dim_x - 1; i++)
//        {
//            idx = row + i;
//            if (barrier[row + i - 1])
//            {
//                f_2[idx] = f_4[row + i - 1];
//            }
//            if (barrier[row + i + 1])
//            {
//                f_4[idx] = f_2[row + i + 1];
//            }
//            if (barrier[rowp + i])
//            {
//                f_1[idx] =   f_3[rowp + i];
//            }
//            if (barrier[rown + i])
//            {
//                f_3[idx] =   f_1[rown + i];
//            }
//            if (barrier[rowp + i - 1])
//            {
//                f_5[idx] =  f_8[rowp + i - 1];
//            }
//            if (barrier[rowp + i + 1])
//            {
//                f_6[idx] =  f_7[rowp + i + 1];
//            }
//            if (barrier[rown + i - 1])
//            {
//                f_7[idx] =  f_6[rown + i - 1];
//            }
//            if (barrier[rown + i + 1])
//            {
//                f_8[idx] =  f_5[rown + i + 1];
//            }
//        }
//    }
//}

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

// check if simulation has become unstable (if so, more time steps are required)
//bool LbmD3Q15::checkStability()
//{
//    int i, idx;
//    bool stable = true;
//
//    for (i = 0; i < dim_x; i++)
//    {
//        idx = (dim_y / 2) * dim_x + i;
//        if (density[idx] <= 0)
//        {
//            stable = false;
//        }
//    }
//    return stable;
//}

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

// compute speed (magnitude of velocity vector)
//void LbmD3Q15::computeSpeed()
//{
//    int i, j, row;
//    for (j = 1; j < dim_y - 1; j++)
//    {
//        row = j * dim_x;
//        for (i = 1; i < dim_x - 1; i++)
//        {
//            speed[row + i] = sqrt(velocity_x[row + i] * velocity_x[row + i] + velocity_y[row + i] * velocity_y[row + i]);
//        }
//    }
//}

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

// compute vorticity (rotational velocity)
//void LbmD3Q15::computeVorticity()
//{
//    int i, j, row, rowp, rown;
//    for (j = 1; j < dim_y - 1; j++)
//    {
//        row = j * dim_x;
//        rowp = (j - 1) * dim_x;
//        rown = (j + 1) * dim_x;
//        for (i = 1; i < dim_x - 1; i++)
//        {
//            vorticity[row + i] = velocity_y[row + i + 1] - velocity_y[row + i - 1] - velocity_x[rown + i] + velocity_x[rowp + i];
//        }
//    }
//}

// gather all data on rank 0
void LbmD3Q15::gatherDataOnRank0(FluidProperty property)
{
    double *send_buf = NULL;
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

// gather all data on rank 0
//void LbmD3Q15::gatherDataOnRank0(FluidProperty property)
//{
//    double *send_buf = NULL;
//    bool *bsend_buf = barrier;
//    switch (property)
//    {
//        case Density:
//            send_buf = density;
//            break;
//        case Speed:
//            send_buf = speed;
//            break;
//        case Vorticity:
//            send_buf = vorticity;
//            break;
//        case None:
//            return;
//    }
//
//    MPI_Status status;
//    MPI_Request request;
//    if (rank == 0)
//    {
//        int i;
//        MPI_Sendrecv(send_buf, 1, own_scalar, 0, 0, recv_buf, 1, other_scalar[0], 0, 0, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(bsend_buf, 1, own_bool, 0, 0, brecv_buf, 1, other_bool[0], 0, 0, MPI_COMM_WORLD, &status);
//        for (i = 1; i < num_ranks; i++)
//        {
//            MPI_Recv(recv_buf, 1, other_scalar[i], i, 0, MPI_COMM_WORLD, &status);
//            MPI_Recv(brecv_buf, 1, other_bool[i], i, 1, MPI_COMM_WORLD, &status);
//        }
//    }
//    else
//    {
//        MPI_Send(send_buf, 1, own_scalar, 0, 0, MPI_COMM_WORLD);
//        MPI_Send(bsend_buf, 1, own_bool, 0, 1, MPI_COMM_WORLD);
//    }
//
//    stored_property = property;
//}

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

// get z start for valid data (0 if no ghost cell on top, 1 if there is a ghost cell on top)uint32_t LbmD3Q15::getStartZ()
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

// get density array
inline double* LbmD3Q15::getDensity()
{
    return density;
}

// get velocity x array
inline double* LbmD3Q15::getVelocityX()
{
    return velocity_x;
}

// get velocity y array
inline double* LbmD3Q15::getVelocityY()
{
    return velocity_y;
}

// get velocity z array
inline double* LbmD3Q15::getVelocityZ()
{
    return velocity_z;
}

// get vorticity array
inline double* LbmD3Q15::getVorticity()
{
    return vorticity;
}

// get vorticity array
inline double* LbmD3Q15::getSpeed()
{
    return speed;
}

// get density array
inline double* LbmD3Q15::getDensity()
{
    if (rank != 0 || stored_property != Density) return NULL;
    return recv_buf;
}

// get vorticity array
inline double* LbmD3Q15::getVorticity()
{
    if (rank != 0 || stored_property != Vorticity) return NULL;
    return recv_buf;
}

// get speed array
inline double* LbmD3Q15::getSpeed()
{
    if (rank != 0 || stored_property != Speed) return NULL;
    return recv_buf;
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

//void LbmD3Q15::setEquilibrium(int x, int y, double new_velocity_x, double new_velocity_y, double new_density)
//{
//    int idx = y * dim_x + x;
//
//    double one_ninth = 1.0 / 9.0;
//    double four_ninths = 4.0 / 9.0;
//    double one_thirtysixth = 1.0 / 36.0;
//
//    double velocity_3x   = 3.0 * new_velocity_x;
//    double velocity_3y   = 3.0 * new_velocity_y;
//    double velocity_x2   = new_velocity_x * new_velocity_x;
//    double velocity_y2   = new_velocity_y * new_velocity_y;
//    double velocity_2xy  = 2.0 * new_velocity_x * new_velocity_y;
//    double vecocity_2    = velocity_x2 + velocity_y2;
//    double vecocity_2_15 = 1.5 * vecocity_2;
//    f_0[idx]  = four_ninths     * new_density * (1.0                                                                 - vecocity_2_15);
//    f_2[idx]  = one_ninth       * new_density * (1.0 + velocity_3x               + 4.5 * velocity_x2                 - vecocity_2_15);
//    f_4[idx]  = one_ninth       * new_density * (1.0 - velocity_3x               + 4.5 * velocity_x2                 - vecocity_2_15);
//    f_1[idx]  = one_ninth       * new_density * (1.0 + velocity_3y               + 4.5 * velocity_y2                 - vecocity_2_15);
//    f_3[idx]  = one_ninth       * new_density * (1.0 - velocity_3y               + 4.5 * velocity_y2                 - vecocity_2_15);
//    f_5[idx] = one_thirtysixth * new_density * (1.0 + velocity_3x + velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15);
//    f_7[idx] = one_thirtysixth * new_density * (1.0 + velocity_3x - velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15);
//    f_6[idx] = one_thirtysixth * new_density * (1.0 - velocity_3x + velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15);
//    f_8[idx] = one_thirtysixth * new_density * (1.0 - velocity_3x - velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15);
//    density[idx]    = new_density;
//    velocity_x[idx] = new_velocity_x;
//    velocity_y[idx] = new_velocity_y;
//}

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

// private - get 2 factors of a given number that are closest to each other
//void LbmD3Q15::getClosestFactors2(int value, int *factor_1, int *factor_2)
//{
//    int test_num = (int)sqrt(value);
//    while (value % test_num != 0)
//    {
//        test_num--;
//    }
//    *factor_2 = test_num;
//    *factor_1 = value / test_num;
//}

// private - exchange boundary information between MPI ranks
void LbmD3Q15::exchangeBoundaries()
{
    MPI_Status status;
    int nx = dim_x;
    int ny = dim_y;
    int sx = start_x;
    int sy = start_y;
    int cx = num_x;
    int cy = num_y;

// f
    MPI_Sendrecv(f, 1, faceN, neighbors[NeighborN], TAG_F,
		 f, 1, faceS, neighbors[NeighborS], TAG_F,
	     	 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(f, 1, faceE, neighbors[NeighborE], TAG_F,                                                              f, 1, faceW, neighbors[NeighborW], TAG_F,                                                              cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(f, 1, faceNE, neighbors[NeighborNE], TAG_F,                                                            f, 1, faceSW, neighbors[NeighborSW], TAG_F,                                                            cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(f, 1, faceNW, neighbors[NeighborNW], TAG_F,                                                            f, 1, faceSE, neighbors[NeighborSE], TAG_F,                                                            cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(f, 1, faceZlo, neighbors[NeighborDown], TAG_F,
		 f, 1, faceZhi, neighbors[NeighborUp], TAG_F,
		 cart_comm, MPI_STATUS_IGNORE);

// density
    MPI_Sendrecv(density, 1, faceN, neighbors[NeighborN], TAG_D,                                                              density, 1, faceS, neighbors[NeighborS], TAG_D,                                                              cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(density, 1, faceE, neighbors[NeighborE], TAG_D,                                                              density, 1, faceW, neighbors[NeighborW], TAG_D,                                                              cart_comm, MPI_STATUS_IGNORE);                                                
    MPI_Sendrecv(density, 1, faceNE, neighbors[NeighborNE], TAG_D,                                                            density, 1, faceSW, neighbors[NeighborSW], TAG_D,                                                            cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(density, 1, faceNW, neighbors[NeighborNW], TAG_D,                                                            density, 1, faceSE, neighbors[NeighborSE], TAG_D,                                                            cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(density, 1, faceZlo, neighbors[NeighborDown], TAG_D,
                 density, 1, faceZhi, neighbors[NeighborUp], TAG_D,
                 cart_comm, MPI_STATUS_IGNORE);

// velocity_x
    MPI_Sendrecv(velocity_x, 1, faceN, neighbors[NeighborN], TAG_VX,                                                              velocity_x, 1, faceS, neighbors[NeighborS], TAG_VX,                                                              cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_x, 1, faceE, neighbors[NeighborE], TAG_VX,                                                              velocity_x, 1, faceW, neighbors[NeighborW], TAG_VX,                                                              cart_comm, MPI_STATUS_IGNORE);                                                                                                                                           
    MPI_Sendrecv(velocity_x, 1, faceNE, neighbors[NeighborNE], TAG_VX,                                                            velocity_x, 1, faceSW, neighbors[NeighborSW], TAG_VX,                                                            cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_x, 1, faceNW, neighbors[NeighborNW], TAG_VX,                                                            velocity_x, 1, faceSE, neighbors[NeighborSE], TAG_VX,                                                            cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_x, 1, faceZlo, neighbors[NeighborDown], TAG_VX,
                 velocity_x, 1, faceZhi, neighbors[NeighborUp], TAG_VX,
                 cart_comm, MPI_STATUS_IGNORE);

// velocity_y
    MPI_Sendrecv(velocity_y, 1, faceN, neighbors[NeighborN], TAG_VY,                                                              velocity_y, 1, faceS, neighbors[NeighborS], TAG_VY,                                                              cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_y, 1, faceE, neighbors[NeighborE], TAG_VY,                                                              velocity_y, 1, faceW, neighbors[NeighborW], TAG_VY,                                                              cart_comm, MPI_STATUS_IGNORE);                                                                                                                                          
    MPI_Sendrecv(velocity_y, 1, faceNE, neighbors[NeighborNE], TAG_VY,                                                            velocity_y, 1, faceSW, neighbors[NeighborSW], TAG_VY,                                                            cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_y, 1, faceNW, neighbors[NeighborNW], TAG_VY,                                                            velocity_y, 1, faceSE, neighbors[NeighborSE], TAG_VY,                                                            cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_y, 1, faceZlo, neighbors[NeighborDown], TAG_VY,
                 velocity_y, 1, faceZhi, neighbors[NeighborUp], TAG_VY,
                 cart_comm, MPI_STATUS_IGNORE);

// velocity_z
    MPI_Sendrecv(velocity_z, 1, faceN, neighbors[NeighborN], TAG_VZ,                                                              velocity_z, 1, faceS, neighbors[NeighborS], TAG_VZ,                                                              cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_z, 1, faceE, neighbors[NeighborE], TAG_VZ,                                                              velocity_z, 1, faceW, neighbors[NeighborW], TAG_VZ,                                                              cart_comm, MPI_STATUS_IGNORE);                                    
    MPI_Sendrecv(velocity_z, 1, faceNE, neighbors[NeighborNE], TAG_VZ,                                                            velocity_z, 1, faceSW, neighbors[NeighborSW], TAG_VZ,                                                            cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_z, 1, faceNW, neighbors[NeighborNW], TAG_VZ,                                                            velocity_z, 1, faceSE, neighbors[NeighborSE], TAG_VZ,                                                            cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(velocity_z, 1, faceZlo, neighbors[NeighborDown], TAG_VZ,
                 velocity_z, 1, faceZhi, neighbors[NeighborUp], TAG_VZ,
                 cart_comm, MPI_STATUS_IGNORE);
}

//    if (neighbors[NeighborN] >= 0)
//    {
//        MPI_Sendrecv(&(f_0[(ny - 2) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_0[(ny - 1) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_1[(ny - 2) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_1[(ny - 1) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_2[(ny - 2) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_2[(ny - 1) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_3[(ny - 2) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_3[(ny - 1) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_4[(ny - 2) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_4[(ny - 1) * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_5[(ny - 2) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_5[(ny - 1) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_6[(ny - 2) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_6[(ny - 1) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_7[(ny - 2) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_7[(ny - 1) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_8[(ny - 2) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_8[(ny - 1) * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(density[(ny - 2) * nx + sx]),    cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(density[(ny - 1) * nx + sx]),    cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_x[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(velocity_x[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_y[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(velocity_y[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
//    }
//    if (neighbors[NeighborE] >= 0)
//    {
//        MPI_Sendrecv(f_0,        1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_0,        1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_1,        1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_1,        1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_2,        1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_2,        1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_3,        1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_3,        1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_4,        1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_4,        1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_5,       1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_5,       1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_6,       1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_6,       1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_7,       1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_7,       1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_8,       1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_8,       1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(density,    1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, density,    1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(velocity_x, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, velocity_x, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(velocity_y, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, velocity_y, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
//    }
//    if (neighbors[NeighborS] >= 0)
//    {
//        MPI_Sendrecv(&(f_0[sy * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_0[sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_1[sy * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_1[sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_2[sy * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_2[sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_3[sy * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_3[sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_4[sy * nx + sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_4[sx]),        cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_5[sy * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_5[sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_6[sy * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_6[sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_7[sy * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_7[sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_8[sy * nx + sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_8[sx]),       cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(density[sy * nx + sx]),    cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(density[sx]),    cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_x[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(velocity_x[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_y[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(velocity_y[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
//    }
//    if (neighbors[NeighborW] >= 0)
//    {
//        MPI_Sendrecv(f_0,        1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_0,        1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_1,        1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_1,        1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_2,        1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_2,        1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_3,        1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_3,        1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_4,        1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_4,        1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_5,       1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_5,       1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_6,       1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_6,       1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_7,       1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_7,       1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(f_8,       1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_8,       1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(density,    1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, density,    1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(velocity_x, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, velocity_x, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(velocity_y, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, velocity_y, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
//    }
//    if (neighbors[NeighborNE] >= 0)
//    {
//        MPI_Sendrecv(&(f_0[(ny - 2) * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_0[(ny - 1) * nx + nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_1[(ny - 2) * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_1[(ny - 1) * nx + nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_2[(ny - 2) * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_2[(ny - 1) * nx + nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_3[(ny - 2) * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_3[(ny - 1) * nx + nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_4[(ny - 2) * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_4[(ny - 1) * nx + nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_5[(ny - 2) * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_5[(ny - 1) * nx + nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_6[(ny - 2) * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_6[(ny - 1) * nx + nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_7[(ny - 2) * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_7[(ny - 1) * nx + nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_8[(ny - 2) * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_8[(ny - 1) * nx + nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(density[(ny - 2) * nx + nx - 2]),    1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(density[(ny - 1) * nx + nx - 1]),    1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_x[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(velocity_x[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_y[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(velocity_y[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
//    }
//    if (neighbors[NeighborNW] >= 0)
//    {
//        MPI_Sendrecv(&(f_0[(ny - 2) * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_0[(ny - 1) * nx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_1[(ny - 2) * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_1[(ny - 1) * nx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_2[(ny - 2) * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_2[(ny - 1) * nx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_3[(ny - 2) * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_3[(ny - 1) * nx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_4[(ny - 2) * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_4[(ny - 1) * nx]),        1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_5[(ny - 2) * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_5[(ny - 1) * nx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_6[(ny - 2) * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_6[(ny - 1) * nx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_7[(ny - 2) * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_7[(ny - 1) * nx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_8[(ny - 2) * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_8[(ny - 1) * nx]),       1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(density[(ny - 2) * nx + sx]),    1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(density[(ny - 1) * nx]),    1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_x[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(velocity_x[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_y[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(velocity_y[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
//    }
//    if (neighbors[NeighborSE] >= 0)
//    {
//        MPI_Sendrecv(&(f_0[sy * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_0[nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_1[sy * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_1[nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_2[sy * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_2[nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_3[sy * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_3[nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_4[sy * nx + nx - 2]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_4[nx - 1]),        1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_5[sy * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_5[nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_6[sy * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_6[nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_7[sy * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_7[nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_8[sy * nx + nx - 2]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_8[nx - 1]),       1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(density[sy * nx + nx - 2]),    1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(density[nx - 1]),    1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_x[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(velocity_x[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_y[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(velocity_y[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
//    }
//    if (neighbors[NeighborSW] >= 0)
//    {
//        MPI_Sendrecv(&(f_0[sy * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_0[0]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_1[sy * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_1[0]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_2[sy * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_2[0]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_3[sy * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_3[0]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_4[sy * nx + sx]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_4[0]),        1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_5[sy * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_5[0]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_6[sy * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_6[0]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_7[sy * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_7[0]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(f_8[sy * nx + sx]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_8[0]),       1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(density[sy * nx + sx]),    1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(density[0]),    1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_x[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(velocity_x[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(&(velocity_y[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(velocity_y[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
//    }

#endif // _LBMD3Q15_MPI_HPP_
