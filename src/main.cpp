#include "../headers/include_lib.h"
#include "../headers/NmrRelaxationSolve.h"

#define NGPU 3

int main(int argc, char **argv)
{
	int MPI_size, MPI_rank, gradientNum[3];
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
	MPI_Barrier(MPI_COMM_WORLD);

	try
	{
		if (argc >= 4)
		{
			gradientNum[0] = atoi(argv[1]);
			gradientNum[1] = atoi(argv[2]);
			gradientNum[2] = atoi(argv[3]);
		}
		else throw 0;
	}
	catch (int a)
	{
		cout << "Caught exception number:  " << a << " gradient pulse failer!"<< endl;
	}
	cout << "Gradient pulses are:   X=" << gradientNum[0] << " Y=" << gradientNum[1] << " Z=" << gradientNum[2] << endl;
	int deviceShift=0;
	if (argc >= 5) deviceShift = atoi(argv[4]);
	printf("Cuda device %d\n", (MPI_rank+deviceShift) % NGPU);
	cudaSetDevice((MPI_rank+deviceShift) % NGPU);

	NmrRelaxationSolve solve_gpu(MPI_size, MPI_rank, gradientNum);	
	solve_gpu.init();
	solve_gpu.solve_gpu();

	MPI_Finalize();
	return 0;
}

