#ifndef PARAVIEW_SIM_HPP
#define PARAVIEW_SIM_HPP

#include <cstdint>
#include <cstdio>
#include <string>
#include <iostream>

class LbmDQ;

void exportSimulationStateToVTK(LbmDQ* lbm, const char* filename, double dt, double dx, double physical_density, uint32_t time_steps) {
    float* gathered = lbm->getGatheredSpeed();
    if (!gathered) {
        fprintf(stderr, "Error: gathered array is NULL!\n");
        return;
    }
    const char* scalar_name = "speed_m_per_s";
    double value_scale = 1.0;
    double speed_scale = dx / dt;
    //printf("[DEBUG] VTK scaling: dx=%.6f, dt=%.6f, speed_scale=%.6f\n", dx, dt, speed_scale);
    //printf("[DEBUG] Expected conversion: 0.0015 lattice -> %.6f physical\n", 0.0015 * speed_scale);
    if (std::string(scalar_name) == "speed_m_per_s") {
        value_scale = speed_scale;
    } else if (std::string(scalar_name) == "density") {
        value_scale = physical_density;
    } else if (std::string(scalar_name) == "vorticity") {
        value_scale = speed_scale / dx;
    }
    int total_x = lbm->getTotalDimX();
    int total_y = lbm->getTotalDimY();
    int total_z = lbm->getTotalDimZ();
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s for writing\n", filename);
        return;
    }
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "LBM Simulation Data (%s) [FIXED_SCALING_v11]\n", scalar_name);
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_GRID\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", total_x, total_y, total_z);
    fprintf(fp, "POINTS %d float\n", total_x * total_y * total_z);
    for (int k = 0; k < total_z; ++k) {
        for (int j = 0; j < total_y; ++j) {
            for (int i = 0; i < total_x; ++i) {
                fprintf(fp, "%d %d %d\n", i, j, k);
            }
        }
    }
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

inline void printVTKDebugInfo(int t, LbmDQ* lbm, float* speed) {
    uint8_t* barrier = lbm->getBarrier();
    int total_x = lbm->getTotalDimX();
    int total_y = lbm->getTotalDimY();
    int total_z = lbm->getTotalDimZ();
    printf("[VTK][t=%d] First 20 nonzero speed, non-barrier nodes (i,j,k): speed, barrier\n", t);
    int found = 0;
    for (int k = 0; k < total_z && found < 20; ++k) {
        for (int j = 0; j < total_y && found < 20; ++j) {
            for (int i = 0; i < total_x && found < 20; ++i) {
                int idx = i + total_x * (j + total_y * k);
                float s = speed ? speed[idx] : -1.0f;
                int b = barrier ? barrier[idx] : -1;
                if (b == 0 && s > 0.0f) {
                    printf("  Node %2d: (%2d,%2d,%2d): speed=%.6f, barrier=%d\n", found, i, j, k, s, b);
                    found++;
                }
            }
        }
    }
    if (found == 0) {
        printf("  No nonzero speed, non-barrier nodes found.\n");
    }
}

inline void printMeanSpeeds(int t, LbmDQ* lbm, float* speed) {
    int N = lbm->getTotalDimX() * lbm->getTotalDimY() * lbm->getTotalDimZ();
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += speed[i];
    printf("[t=%d] Mean speed: %.8f\n", t, sum / N);
    uint8_t* barrier = lbm->getBarrier();
    int total_x = lbm->getTotalDimX();
    int total_y = lbm->getTotalDimY();
    int total_z = lbm->getTotalDimZ();
    double sum_plane = 0.0;
    int count_plane = 0;
    for (int k = 0; k < total_z; ++k) {
        for (int j = 0; j < total_y; ++j) {
            int idx = 1 + total_x * (j + total_y * k);
            if (barrier && barrier[idx]) continue;
            sum_plane += speed[idx];
            count_plane++;
        }
    }
}

inline void printSpeedProfileX(int t, LbmDQ* lbm, float* speed) {
    int total_x = lbm->getTotalDimX();
    int total_y = lbm->getTotalDimY();
    int total_z = lbm->getTotalDimZ();
    int j = total_y / 2;
    int k = total_z / 2;
    printf("[t=%d] Speed profile along x at y=%d, z=%d:\n", t, j, k);
    for (int i = 0; i < total_x; ++i) {
        int idx = i + total_x * (j + total_y * k);
        printf("  x=%2d: %.6f\n", i, speed[idx]);
    }
}

inline void printSimulationDiagnostics(int t, int rank, LbmDQ* lbm, double dt, double dx, double physical_density, uint32_t time_steps) {
    if (t % 1000 == 0 && t <= 5000) {
	lbm->computeSpeed();

        lbm->gatherDataOnRank0(LbmDQ::Speed);
        
        if (rank == 0) {
            char vtk_filename[128];
            snprintf(vtk_filename, sizeof(vtk_filename), "simulation_state_t%05d.vtk", t);
            exportSimulationStateToVTK(lbm, vtk_filename, dt, dx, physical_density, time_steps);
            float* speed = lbm->getGatheredSpeed();
        }
    }
    if (t % 1000 == 0 && t <= 5000) {
	lbm->computeSpeed();
        lbm->gatherDataOnRank0(LbmDQ::Speed);
        if (rank == 0) {
            float* speed = lbm->getGatheredSpeed();
            printMeanSpeeds(t, lbm, speed);
        }
    }
}

#endif // PARAVIEW_SIM_HPP 
