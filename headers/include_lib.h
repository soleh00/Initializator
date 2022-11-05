#ifndef include_libh
#define include_libh
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
//#include <conio.h>
#include <ctime>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <iterator>
//#include <direct.h>
#include <mpi.h>
using namespace std;

//#define NPAHSES 2
extern int MPI_size, MPI_rank;

struct GridSize { int ik; int jk; int kk; int num; };
struct VectorField { double *x; double *y; double *z; };
struct RelaxationTimeField { double *t1; double *t2; };
#endif


