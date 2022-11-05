#ifndef NmrRelaxationSolveh
#define NmrRelaxationSolveh
#include "include_lib.h"

#ifndef gpuasserth
#define gpuasserth

#include <cuda_runtime.h>
#include <string>
#include <sstream>

#define gpuErrchk(ans) { gpuAssert ((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::string tmpstr = cudaGetErrorString(code);
		std::ostringstream ss;
		ss << std::dec << line;
		tmpstr = "GPUassert:" + tmpstr + file + ss.str();
		ss.clear();
		std::cout << tmpstr << std::endl << std::endl;
		//fprintf(stderr,"GPUassert: %s %s %d\n",cudaGetErrorString(code), file,line);
		if (abort) exit(code);
	}
};

#endif


class NmrRelaxationSolve
{
private:
	// global structures/variables for each cell
	GridSize gs, gsBC;
	double *concentration[2], *diffusion, *concentrationBC[2], *diffusionBC;
	VectorField mainField, noiseField, velocity, magnetization, addMagnetization, surfaceRel1Water, surfaceRel2Water, surfaceRel1Oil, surfaceRel2Oil;
	RelaxationTimeField relTime;
	char *actCell;
	// general variables
		//read from nmr_start.dat
	int key,iPrint,jPrint,kPrint,ndPrint,ndWrite,ndVid,ndAmpl,nStep;
	int nPoro, nComponents, nFluid, nVectorMult, nVelocity, nDiffusion, nDirectionField, nSequence, nSurfRelWater, nSurfRelOil;
	int nBoundary1, nBoundary2, nBoundary3, nBoundary4, nBoundary5, nBoundary6;
	int flag1, flag2, flag3, flag4, flag5;
	int nnx, nny, nnz;
	int bufferSize, bufferSizeX, bufferSizeY, bufferSizeZ;
	double dx, dy, dz, bsx, bsy, bsz, sequenceTime1, sequenceTime2, gradientPulse, zeroAmplitude, velocityMultiplicator, direction[3], velocityAverage, multipleGradients[3];
	double constGradX, constGradY, constGradZ, noiseAmpl;
	double dTimeVel, dTimeDiff, dTimeRel, dTimeNoise, dTimeGrad, dTime, currentTime, endTime;
	double dOil, dWater, r1Oil, r2Oil, r1Water, r2Water, t1Water, t1Oil, t2Water, t2Oil;
	double an1, an2, bn1, bn2, temperature, cfl, cfl1, step0, scale;
	double bolchman, plank, avogadro, pi, g;
	double big, small;
	double equilibriumMagnetization, results, resX, resY, resZ, resX1, resY1, resZ1, resX2, resY2, resZ2;
	char* actCell_d;
	double* diff_d, *magnx_d, *magny_d, *magnz_d,
		*mFx_d, *mFy_d, *mFz_d, *nFx_d, *nFy_d, *nFz_d,
		*velx_d, *vely_d, *velz_d,
		*addMagx_d, *addMagy_d, *addMagz_d,
		*predictorX, *predictorY, *predictorZ,
		*correctorX, *correctorY, *correctorZ,
		*softnerX, *softnerY, *softnerZ,
		*relT1_d, *relT2_d;

	// private methods
	void ReadGridSize();
	void ReadAndFillInAll();
	void ReadStart();
	void ReadActiveCells();
	void BoundaryConditionsForActiveCells();
	void BoundaryConditionsForDiffusion();
	void BoundaryConditionsForVectorField(VectorField* field);
	void BoundaryConditionsForVectorField_gpu(double* magnx_d, double* magny_d, double* magnz_d, VectorField* magn, GridSize gs);
	void ReadConcentration();
	void ReadVelocity();
	void FillInVelocityBC();
	void ReadSurfRel();
	void FillInDiffusinAndTime();
	void FillInMainField();
	void FillInMagnetization();
	void StepTime();
	void ExplicitSimple();
	void ExplicitMaccormac();
	void ExplicitMaccormacUpgrade();
	void InitializeGpu();

	//make it external
	void ExplicitSimpleCuda();
	void OrtTransform(int i,int j,int k, double nx, double ny, double nz, VectorField* init, double angle);
	void AddImpulse(int flag, double grad, double nx, double ny, double nz, VectorField* init);
	void Sequence();
	void Sequence_gpu(double* ,double* ,double* );
	// debug print
	void DebugPrint();
	void ExplicitMakUpgrade();


	void PrintActiveCells();
	void PrintConcentration();
	void PrintVectorField(int flag,VectorField* field);
	void WriteVectorField(int flag, VectorField* field, string fileName);
	void PrintDiffusionAndRelTime();
	void OutputX(VectorField* field), OutputY(VectorField* field), OutputZ(VectorField* field);
	void WriteVectorFieldSeparated(VectorField* field, string fileNamePattern);


	void AllocateAll();
	void AllocateAllBC();
	void InitCudaArrays();
	void ClearAll();

	int index(int i, int j, int k, GridSize gs);
	double min(double a, double b);
	double max(double a, double b);
public:
	NmrRelaxationSolve(int size, int rank, int *num)
	{
		MPI_size = size;
		MPI_rank = rank;
		PulseNum[0] = num[0];
		PulseNum[1] = num[1];
		PulseNum[2] = num[2];
	}
	int MPI_size, MPI_rank, PulseNum[3];
	void solve();
	void init();
	void solve_gpu();
};

#endif