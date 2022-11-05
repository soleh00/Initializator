#include "../headers/NmrRelaxationSolve.h"
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <iomanip>
#include <mpi.h>
#include <cstring>
#include <math.h>
#include <cuda_runtime.h>
#include "../headers/NmrRelaxationSolveKernels.cuh"

//#include <stdlib.h>
//-----------------------------------------------------------------------

//-----------------------------------------------------------------


void NmrRelaxationSolve::init()
{
	ReadGridSize();
	AllocateAll();
	ReadAndFillInAll();
	/*if ((nVelocity == 1) &&  (nBoundary1 == 2 || nBoundary2 == 2 || nBoundary3 == 2 || nBoundary4 == 2 || nBoundary5 == 2 || nBoundary6 == 2))
	{
		ReadGridSizeBC();
		AllocateAllBC();
		FillInAllBC();
	}*/
	//DebugPrint();
		
}

void NmrRelaxationSolve::solve()
{
	ofstream res("results_cpu.csv");
	MPI_Barrier(MPI_COMM_WORLD);
	if (MPI_rank == 0) res << "Time; SumX; SumY; SumZ" << endl;	
	for (int i = 0; i < nStep + 1; i++)
	{
		currentTime = i*dTime;
		if ((i % ndPrint) == 0)
		{
			OutputX(&magnetization);
			OutputY(&magnetization);
			OutputZ(&magnetization);
			if (MPI_rank == 0)
			{
				cout << " Time = " << currentTime;
				cout << " SumX=" << resX << " SumY=" << resY << " SumZ=" << resZ << endl;
				res << scientific << currentTime << "; " << scientific << resX << "; " << scientific << resY << "; " << scientific << resZ << endl;
			}
			/*Test field write*/
			//WriteVectorField(0, &magnetization, (string("cpu") + std::to_string(i) + string(".dat")));
		}
		ExplicitSimple();
		Sequence();
		BoundaryConditionsForVectorField(&magnetization);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if (MPI_rank == 0) res.close();


	ClearAll();
}

void NmrRelaxationSolve::ReadGridSize()
{
	std::string field_comment;
	ifstream f("data/Nmr_start.dat");
	try
	{
		if (!f) throw 1;
	}
	catch (int a)
	{
		cout << "Caught exception number:  " << a << " No file Nmr_start.dat" << endl;
	}

	f >> gs.ik;
	getline(f, field_comment);
	f >> gs.jk;
	getline(f, field_comment);
	f >> gs.kk;
	getline(f, field_comment);
	gs.kk = gs.kk / MPI_size;
	cout << "X grid size" << gs.ik << endl;
	cout << "Y grid size" << gs.jk << endl;
	cout << "Z grid size" << gs.kk << endl;
	gs.num = (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2);
	f.close();
}


void NmrRelaxationSolve::AllocateAll()
{
	for (int i = 0; i < 2; i++)
	{
		concentration[i] = new double[gs.num];
	}
	diffusion = new double[gs.num];
	mainField.x = new double[gs.num];
	mainField.y = new double[gs.num];
	mainField.z = new double[gs.num];
	noiseField.x = new double[gs.num];
	noiseField.y = new double[gs.num];
	noiseField.z = new double[gs.num];
	velocity.x = new double[gs.num];
	velocity.y = new double[gs.num];
	velocity.z = new double[gs.num];
	relTime.t1 = new double[gs.num];
	relTime.t2 = new double[gs.num];
	magnetization.x = new double[gs.num];
	magnetization.y = new double[gs.num];
	magnetization.z = new double[gs.num];
	addMagnetization.x = new double[gs.num];
	addMagnetization.y = new double[gs.num];
	addMagnetization.z = new double[gs.num];
	surfaceRel1Water.x = new double[gs.num];
	surfaceRel1Water.y = new double[gs.num];
	surfaceRel1Water.z = new double[gs.num];
	surfaceRel2Water.x = new double[gs.num];
	surfaceRel2Water.y = new double[gs.num];
	surfaceRel2Water.z = new double[gs.num];
	surfaceRel1Oil.x = new double[gs.num];
	surfaceRel1Oil.y = new double[gs.num];
	surfaceRel1Oil.z = new double[gs.num];
	surfaceRel2Oil.x = new double[gs.num];
	surfaceRel2Oil.y = new double[gs.num];
	surfaceRel2Oil.z = new double[gs.num];
	actCell = new char[gs.num];

}

void NmrRelaxationSolve::ClearAll()
{
	for (int i = 0; i < 2; i++)
	{
		delete[] concentration[i];
	}
	delete[] diffusion;
	delete[] mainField.x;
	delete[] mainField.y;
	delete[] mainField.z;
	delete[] noiseField.x;
	delete[] noiseField.y;
	delete[] noiseField.z;
	delete[] velocity.x;
	delete[] velocity.y;
	delete[] velocity.z;
	delete[] relTime.t1;
	delete[] relTime.t2;
	delete[] magnetization.x;
	delete[] magnetization.y;
	delete[] magnetization.z;
	delete[] addMagnetization.x;
	delete[] addMagnetization.y;
	delete[] addMagnetization.z;
	delete[] surfaceRel1Water.x;
	delete[] surfaceRel1Water.y;
	delete[] surfaceRel1Water.z;
	delete[] surfaceRel2Water.x;
	delete[] surfaceRel2Water.y;
	delete[] surfaceRel2Water.z;
	delete[] surfaceRel1Oil.x;
	delete[] surfaceRel1Oil.y;
	delete[] surfaceRel1Oil.z;
	delete[] surfaceRel2Oil.x;
	delete[] surfaceRel2Oil.y;
	delete[] surfaceRel2Oil.z;
	delete[] actCell;
}

void NmrRelaxationSolve::ReadStart()
{
	std::string field_comment;
	ifstream f("data/Nmr_data.dat");
	try
	{
		if (!f) throw 1;
	}
	catch (int a)
	{
		cout << "Caught exception number:  " << a << " No file Nmr_data.dat" << endl;
	}
	std::vector<std::pair<std::string, std::string > > keys;
	std::string tmp1, tmp2;
	for(int i = 0; i < 61; ++i)
	{
		std::getline(f, tmp1, '	');
		std::getline(f, tmp2, '\n');
		keys.push_back(make_pair(tmp1, tmp2));
	}
	f.close();
	/*for(auto it : keys)
	{
		std::cout << it.first << "\t" << it.second << std::endl;
	}*/

	for(auto it : keys)
	{
		size_t pos = it.second.find("KEY");
    	if (pos != std::string::npos)
    	{
        	key = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("IPRINT");
    	if (pos != std::string::npos)
    	{
        	iPrint = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("JPRINT");
    	if (pos != std::string::npos)
    	{
        	jPrint = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("KPRINT");
    	if (pos != std::string::npos)
    	{
        	kPrint = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("Printing on");
    	if (pos != std::string::npos)
    	{
        	ndPrint = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NDWRIT");
    	if (pos != std::string::npos)
    	{
        	ndWrite = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NDVID");
    	if (pos != std::string::npos)
    	{
        	ndVid = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NDAMPL");
    	if (pos != std::string::npos)
    	{
        	ndAmpl = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NPOR");
    	if (pos != std::string::npos)
    	{
        	nPoro = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NCOMP");
    	if (pos != std::string::npos)
    	{
        	nComponents = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NFLUID");
    	if (pos != std::string::npos)
    	{
        	nFluid = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("nSurfRelWater");
    	if (pos != std::string::npos)
    	{
        	nSurfRelWater = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("nSurfRelOil");
    	if (pos != std::string::npos)
    	{
        	nSurfRelOil = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NVECTOR");
    	if (pos != std::string::npos)
    	{
        	nVectorMult = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NVELOC");
    	if (pos != std::string::npos)
    	{
        	nVelocity = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NDIFFUS");
    	if (pos != std::string::npos)
    	{
        	nDiffusion = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NDIRECT_FIELD");
    	if (pos != std::string::npos)
    	{
        	nDirectionField = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NSEQUENCE");
    	if (pos != std::string::npos)
    	{
        	nSequence = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NB1");
    	if (pos != std::string::npos)
    	{
        	nBoundary1 = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NB2");
    	if (pos != std::string::npos)
    	{
        	nBoundary2 = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NB3");
    	if (pos != std::string::npos)
    	{
        	nBoundary3 = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NB4");
    	if (pos != std::string::npos)
    	{
        	nBoundary4 = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NB5");
    	if (pos != std::string::npos)
    	{
        	nBoundary5 = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("NB6");
    	if (pos != std::string::npos)
    	{
        	nBoundary6 = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("DX");
    	if (pos != std::string::npos)
    	{
        	dx = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("DY");
    	if (pos != std::string::npos)
    	{
        	dy = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("DZ");
    	if (pos != std::string::npos)
    	{
        	dz = atof(it.first.c_str());
        	break;
    	}
	}

	for(auto it : keys)
	{
		size_t pos = it.second.find("BX0");
    	if (pos != std::string::npos)
    	{
        	bsx = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("BY0");
    	if (pos != std::string::npos)
    	{
        	bsy = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("BZ0");
    	if (pos != std::string::npos)
    	{
        	bsz = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("TIME1");
    	if (pos != std::string::npos)
    	{
        	sequenceTime1 = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("TIME2");
    	if (pos != std::string::npos)
    	{
        	sequenceTime2 = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("GRAD_X");
    	if (pos != std::string::npos)
    	{
        	constGradX = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("GRAD_Y");
    	if (pos != std::string::npos)
    	{
        	constGradY = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("GRAD_Z");
    	if (pos != std::string::npos)
    	{
        	constGradZ = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("AMPL_NOISE");
    	if (pos != std::string::npos)
    	{
        	noiseAmpl = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("VEL_MULT");
    	if (pos != std::string::npos)
    	{
        	velocityMultiplicator = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("D_OIL");
    	if (pos != std::string::npos)
    	{
        	dOil = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("D_WAT");
    	if (pos != std::string::npos)
    	{
        	dWater = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("R1_OIL");
    	if (pos != std::string::npos)
    	{
        	r1Oil = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("R1_WAT");
    	if (pos != std::string::npos)
    	{
        	r1Water = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("R2_OIL");
    	if (pos != std::string::npos)
    	{
        	r2Oil = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("R2_WAT");
    	if (pos != std::string::npos)
    	{
        	r2Water = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("T1_OIL");
    	if (pos != std::string::npos)
    	{
        	t1Oil = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("T1_WAT");
    	if (pos != std::string::npos)
    	{
        	t1Water = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("T2_OIL");
    	if (pos != std::string::npos)
    	{
        	t2Oil = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("T2_WAT");
    	if (pos != std::string::npos)
    	{
        	t2Water = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("T0");
    	if (pos != std::string::npos)
    	{
        	temperature = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("AN(1)");
    	if (pos != std::string::npos)
    	{
        	an1 = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("AN(2)");
    	if (pos != std::string::npos)
    	{
        	an2 = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("BN(1)");
    	if (pos != std::string::npos)
    	{
        	bn1 = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("BN(2)");
    	if (pos != std::string::npos)
    	{
        	bn2 = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("CFL");
    	if (pos != std::string::npos)
    	{
        	cfl = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("CFL1");
    	if (pos != std::string::npos)
    	{
        	cfl1 = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("TIMEW");
    	if (pos != std::string::npos)
    	{
        	step0 = atof(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("bufferX");
    	if (pos != std::string::npos)
    	{
        	bufferSizeX = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("bufferY");
    	if (pos != std::string::npos)
    	{
        	bufferSizeY = atoi(it.first.c_str());
        	break;
    	}
	}
	for(auto it : keys)
	{
		size_t pos = it.second.find("bufferZ");
    	if (pos != std::string::npos)
    	{
        	bufferSizeZ = atoi(it.first.c_str());
        	break;
    	}
	}

	//��������, ������� �������� � ������ �������� ������ �����
	cout << endl << "Countable parametres" << endl;
	ifstream gradient("data/Nmr_grad.dat");
	try
	{
		if (!gradient) throw 1;
	}
	catch (int a)
	{
		cout << "Caught exception number:  " << a << " No file Nmr_grad.dat" << endl;
	}
	int total;
	gradient >> total;
	cout << "Total = " << total << endl;
	getline(gradient, field_comment);
	if (total < PulseNum[0]) exit(1);
	for (int i = 0; i < PulseNum[0] - 1; i++) getline(gradient, field_comment);
	gradient >> gradientPulse;
	gradient.close();

	ifstream gradientX("data/Nmr_grad.dat");
	if (PulseNum[0] != 0)
	{
		for (int i = 0; i < PulseNum[0]; i++) getline(gradientX, field_comment);
		gradientX >> multipleGradients[0];
	}
	else
	{
		multipleGradients[0] = 0.0;
	}
	gradientX.close();
	ifstream gradientY("data/Nmr_grad.dat");
	if (PulseNum[1] != 0)
	{
		for (int i = 0; i < PulseNum[1]; i++) getline(gradientY, field_comment);
		gradientY >> multipleGradients[1];
	}
	else
	{
		multipleGradients[1] = 0.0;
	}
	gradientY.close();
	ifstream gradientZ("data/Nmr_grad.dat");
	if (PulseNum[2] != 0)
	{
		for (int i = 0; i < PulseNum[2]; i++) getline(gradientZ, field_comment);
		gradientZ >> multipleGradients[2];
	}
	else
	{
		multipleGradients[2] = 0.0;
	}

	gradientZ.close();
	//�������� �����������, �� �� ������� � �����.
	double b_total = sqrt(bsx*bsx + bsy*bsy + bsz*bsz);
	if (nSequence == 0) endTime = 2 * t2Water;
	if (nSequence == 1) endTime = 2 * t2Water;
	if (nSequence == 2) endTime = 4.0*sequenceTime1 + sequenceTime2;
	if (nSequence == 5) endTime = sequenceTime2 + 2.0*sequenceTime1;
	if (nSequence == 9 || nSequence == 99 || nSequence == 98) endTime = sequenceTime2 + 2.0*sequenceTime1;
	for (int i = 0; i < 3; i++)
	{
		direction[i] = 0.0;
		if (nDirectionField == i + 1) direction[i] = 1.0;
		if (nDirectionField == 0)
		{
			if (i == 0) direction[0] = bsx / b_total;
			if (i == 1) direction[1] = bsy / b_total;
			if (i == 2) direction[2] = bsz / b_total;
		}
		cout << "direction["<<i+1<<"]=" << direction[i] << endl;
	}
	flag1 = 0;
	flag2 = 0;
	flag3 = 0;
	flag4 = 0;
	flag5 = 0;
	bolchman = 1.380662e-23;
	plank = 1.0545887e-34;
	avogadro = 6.022e+23; 
	pi = 3.1415926535897932384626433832795028841971;
	g = 4.78941714e+7;
	big = 1.0e+50;
	small = 1.0e-50;
	equilibriumMagnetization = 1.0 /double ((gs.ik - 2*bufferSizeX)*(gs.jk - 2*bufferSizeY)*(gs.kk - 2*bufferSizeZ));//1.0;//tanh(0.5*g*b_total*plank / (temperature*bolchman));
	cout << "endTime=" << endTime << endl;
	cout << "gradientPulse=" << gradientPulse << endl;
	cout << endl << "Gradients for X = " << multipleGradients[0] << " Y = " << multipleGradients[1] << " Z = " << multipleGradients[2] << endl << endl;
	cout << "equilibriumMagnetization=" << scientific << setprecision(15) << equilibriumMagnetization << endl;
	cout << "gs.kk=" << gs.kk << " gs.num=" << gs.num << " MPI_rank=" << MPI_rank << endl;
	MPI_Barrier(MPI_COMM_WORLD);
}

void NmrRelaxationSolve::ReadActiveCells()
{
	//0 ���  ���� �����, ���� �� ���
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				actCell[index(i, j, k, gs)] = '0';
			}
		}
	}
	if (nPoro == 0)
	{
		for (int k = 1; k < (gs.kk + 1); k++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int i = 1; i < (gs.ik + 1); i++)
				{
					actCell[index(i, j, k, gs)] = '1';
				}
			}
		}
	}
	
	if (nPoro == 1)
	{
		char buf;
		std::string field_comment;
		ifstream f("data/Poro.dat");
		try
		{
			if (!f) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file Poro.dat" << endl;
		}
		getline(f, field_comment);
		for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int i = 1; i < (gs.ik + 1); i++)
				{
					f >> buf;
					if (int((k-1)/ gs.kk) == MPI_rank)
					{
						actCell[index(i, j, ((k-1) % gs.kk+1), gs)]=buf;
					}
				}
			}
		}
		f.close();
	}
	cout << "Poro read successfully!" << endl;
	MPI_Barrier(MPI_COMM_WORLD);
}

void NmrRelaxationSolve::BoundaryConditionsForActiveCells()
{
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			actCell[index(0, j, k, gs)] = actCell[index(1, j, k, gs)];
			actCell[index(gs.ik + 1, j, k, gs)] = actCell[index(gs.ik, j, k, gs)];
		}
	}
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int i = 0; i < (gs.ik + 2); i++)
		{
			actCell[index(i, 0, k, gs)] = actCell[index(i, 1, k, gs)];
			actCell[index(i, gs.jk+1, k, gs)] = actCell[index(i, gs.jk, k, gs)];
		}
	}
	//MPI boundary conditions
	MPI_Status status;
	MPI_Sendrecv(&actCell[index(0, 0, gs.kk, gs)], (gs.ik+2)*(gs.jk+2), MPI_CHAR, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
		&actCell[index(0, 0, 0, gs)], (gs.ik + 2)*(gs.jk + 2), MPI_CHAR, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
		MPI_COMM_WORLD, &status);
	MPI_Sendrecv(&actCell[index(0, 0, 1, gs)], (gs.ik + 2)*(gs.jk + 2), MPI_CHAR, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
		&actCell[index(0, 0, gs.kk + 1, gs)], (gs.ik + 2)*(gs.jk + 2), MPI_CHAR, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
		MPI_COMM_WORLD, &status);
	//end of MPI boundary conditions. Corrections within 0 & size-1 ranks
	if (MPI_rank == 0)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				actCell[index(i, j, 0, gs)] = actCell[index(i, j, 1, gs)];
			}
		}
	}
	if (MPI_rank == (MPI_size - 1))
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				actCell[index(i, j, gs.kk + 1, gs)] = actCell[index(i, j, gs.kk, gs)];
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void NmrRelaxationSolve::ReadConcentration()
{
	for (int nPhase = 0; nPhase < 2; nPhase++)
	{
		//���������
		for (int k = 0; k < (gs.kk + 2); k++)
		{
			for (int j = 0; j < (gs.jk + 2); j++)
			{
				for (int i = 0; i < (gs.ik + 2); i++)
				{
					concentration[nPhase][index(i, j, k, gs)] = 0.0;
				}
			}
		}
		if (nComponents == 1)
		{
			if (nFluid == 1)
			{
				for (int k = 1; k < (gs.kk + 1); k++)
				{
					for (int j = 1; j < (gs.jk + 1); j++)
					{
						for (int i = 1; i < (gs.ik + 1); i++)
						{
							concentration[0][index(i, j, k, gs)] = 1.0;
						}
					}
				}
			}
			if (nFluid == 2)
			{
				for (int k = 1; k < (gs.kk + 1); k++)
				{
					for (int j = 1; j < (gs.jk + 1); j++)
					{
						for (int i = 1; i < (gs.ik + 1); i++)
						{
							concentration[1][index(i, j, k, gs)] = 1.0;
						}
					}
				}
			}
		}
		if (nComponents == 2)
		{
			float buf;
			//printf("Read the poro file plz\n");
			std::string field_comment;
			ifstream f1("data/n1_init.dat");
			try
			{
				if (!f1) throw 1;
			}
			catch (int a)
			{
				cout << "Caught exception number:  " << a << " No file n1_init.dat" << endl;
			}
			getline(f1, field_comment);
			for (int i = 1; i < (gs.ik + 1); i++)
			{
				for (int j = 1; j < (gs.jk + 1); j++)
				{
					for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
					{
						f1 >> buf;
						if (int((k - 1) / gs.kk) == MPI_rank)
						{
							concentration[0][index(i, j, ((k-1) % gs.kk + 1), gs)] = buf;
						}
						if (int((k - 1) / gs.kk) == MPI_rank)
						{
							concentration[1][index(i, j, ((k-1) % gs.kk + 1), gs)] = (1.0 - buf);
						}
					}
				}
			}
			f1.close();
		}
	}
}

void NmrRelaxationSolve::ReadSurfRel()
{
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				surfaceRel1Water.x[index(i, j, k, gs)] = 0.0;
				surfaceRel1Water.y[index(i, j, k, gs)] = 0.0;
				surfaceRel1Water.z[index(i, j, k, gs)] = 0.0;
				surfaceRel2Water.x[index(i, j, k, gs)] = 0.0;
				surfaceRel2Water.y[index(i, j, k, gs)] = 0.0;
				surfaceRel2Water.z[index(i, j, k, gs)] = 0.0;
				surfaceRel1Oil.x[index(i, j, k, gs)] = 0.0;
				surfaceRel1Oil.y[index(i, j, k, gs)] = 0.0;
				surfaceRel1Oil.z[index(i, j, k, gs)] = 0.0;
				surfaceRel2Oil.x[index(i, j, k, gs)] = 0.0;
				surfaceRel2Oil.y[index(i, j, k, gs)] = 0.0;
				surfaceRel2Oil.z[index(i, j, k, gs)] = 0.0;
			}
		}
	}
	if (nSurfRelWater == 1)
	{
		std::string field_comment;
		double buf;
		//Now surface relaxation 1
		ifstream fwx1("data/SurfRel1WaterX.dat");
		try
		{
			if (!fwx1) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel1WaterX.dat" << endl;
		}
		getline(fwx1, field_comment);
		for (int i = 1; i < (gs.ik + 2); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
				{
					fwx1 >> buf;
					if (int((k - 1) / gs.kk) == MPI_rank)
					{
						surfaceRel1Water.x[index(i, j, ((k - 1) % gs.kk + 1), gs)] = buf;
					}
				}
			}
		}
		fwx1.close();
		cout << "surfaceRel1WaterX!" << endl;
		ifstream fwy1("data/SurfRel1WaterY.dat");
		try
		{
			if (!fwy1) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel1WaterY.dat" << endl;
		}
		getline(fwy1, field_comment);
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 2); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
				{
					fwy1 >> buf;
					if (int((k - 1) / gs.kk) == MPI_rank)
					{
						surfaceRel1Water.y[index(i, j, ((k - 1) % gs.kk + 1), gs)] = buf;
					}
				}
			}
		}
		fwy1.close();
		cout << "SurfRel1WaterY done!" << endl;
		ifstream fwz1("data/SurfRel1WaterZ.dat");
		try
		{
			if (!fwz1) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel1WaterZ.dat" << endl;
		}
		getline(fwz1, field_comment);
		double *bufArray;
		bufArray = new double[gs.num*(gs.kk*MPI_size + 2) / (gs.kk + 2)];
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 2); k++)
				{
					fwz1 >> buf;
					bufArray[index(i, j, k, gs)] = buf;
				}
			}
		}
		fwz1.close();
		cout << "SurfRel1WaterZ done!" << endl;
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk + 2); k++)
				{
					surfaceRel1Water.z[index(i, j, k, gs)] = bufArray[index(i, j, k + MPI_rank*gs.kk, gs)];
				}
			}
		}
		//Now surface relaxation 2
		ifstream fwx2("data/SurfRel2WaterX.dat");
		try
		{
			if (!fwx2) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel2WaterX.dat" << endl;
		}
		getline(fwx2, field_comment);
		for (int i = 1; i < (gs.ik + 2); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
				{
					fwx2 >> buf;
					if (int((k - 1) / gs.kk) == MPI_rank)
					{
						surfaceRel2Water.x[index(i, j, ((k - 1) % gs.kk + 1), gs)] = buf;
					}
				}
			}
		}
		fwx2.close();
		cout << "surfaceRel2WaterX!" << endl;
		ifstream fwy2("data/SurfRel2WaterY.dat");
		try
		{
			if (!fwy2) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel2WaterY.dat" << endl;
		}
		getline(fwy2, field_comment);
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 2); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
				{
					fwy2 >> buf;
					if (int((k - 1) / gs.kk) == MPI_rank)
					{
						surfaceRel2Water.y[index(i, j, ((k - 1) % gs.kk + 1), gs)] = buf;
					}
				}
			}
		}
		fwy2.close();
		cout << "SurfRel2WaterY done!" << endl;
		ifstream fwz2("data/SurfRel2WaterZ.dat");
		try
		{
			if (!fwz2) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel2WaterZ.dat" << endl;
		}
		getline(fwz2, field_comment);
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 2); k++)
				{
					fwz2 >> buf;
					bufArray[index(i, j, k, gs)] = buf;
				}
			}
		}
		fwz2.close();
		cout << "SurfRel2WaterZ done!" << endl;
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk + 2); k++)
				{
					surfaceRel2Water.z[index(i, j, k, gs)] = bufArray[index(i, j, k + MPI_rank*gs.kk, gs)];
				}
			}
		}
		delete[] bufArray;
	}
	else
	{
		for (int k = 0; k < (gs.kk + 2); k++)
		{
			for (int j = 0; j < (gs.jk + 2); j++)
			{
				for (int i = 0; i < (gs.ik + 2); i++)
				{
					surfaceRel1Water.x[index(i, j, k, gs)] = r1Water;
					surfaceRel1Water.y[index(i, j, k, gs)] = r1Water;
					surfaceRel1Water.z[index(i, j, k, gs)] = r1Water;
					surfaceRel2Water.x[index(i, j, k, gs)] = r2Water;
					surfaceRel2Water.y[index(i, j, k, gs)] = r2Water;
					surfaceRel2Water.z[index(i, j, k, gs)] = r2Water;
				}
			}
		}
	}


	if (nSurfRelOil == 1)
	{
		std::string field_comment;
		double buf;
		//Now surface relaxation 1
		ifstream fox1("data/SurfRel1OilX.dat");
		try
		{
			if (!fox1) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel1OilX.dat" << endl;
		}
		getline(fox1, field_comment);
		for (int i = 1; i < (gs.ik + 2); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
				{
					fox1 >> buf;
					if (int((k - 1) / gs.kk) == MPI_rank)
					{
						surfaceRel1Oil.x[index(i, j, ((k - 1) % gs.kk + 1), gs)] = buf;
					}
				}
			}
		}
		fox1.close();
		cout << "surfaceRel1OilX!" << endl;
		ifstream foy1("data/SurfRel1OilY.dat");
		try
		{
			if (!foy1) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel1OilY.dat" << endl;
		}
		getline(foy1, field_comment);
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 2); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
				{
					foy1 >> buf;
					if (int((k - 1) / gs.kk) == MPI_rank)
					{
						surfaceRel1Oil.y[index(i, j, ((k - 1) % gs.kk + 1), gs)] = buf;
					}
				}
			}
		}
		foy1.close();
		cout << "SurfRel1OilY done!" << endl;
		ifstream foz1("data/SurfRel1OilZ.dat");
		try
		{
			if (!foz1) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel1OilZ.dat" << endl;
		}
		getline(foz1, field_comment);
		double *bufArray;
		bufArray = new double[gs.num*(gs.kk*MPI_size + 2) / (gs.kk + 2)];
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 2); k++)
				{
					foz1 >> buf;
					bufArray[index(i, j, k, gs)] = buf;
				}
			}
		}
		foz1.close();
		cout << "SurfRel1OilZ done!" << endl;
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk + 2); k++)
				{
					surfaceRel1Oil.z[index(i, j, k, gs)] = bufArray[index(i, j, k + MPI_rank*gs.kk, gs)];
				}
			}
		}
		//Now surface relaxation 2
		ifstream fox2("data/SurfRel2OilX.dat");
		try
		{
			if (!fox2) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel2OilX.dat" << endl;
		}
		getline(fox2, field_comment);
		for (int i = 1; i < (gs.ik + 2); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
				{
					fox2 >> buf;
					if (int((k - 1) / gs.kk) == MPI_rank)
					{
						surfaceRel2Oil.x[index(i, j, ((k - 1) % gs.kk + 1), gs)] = buf;
					}
				}
			}
		}
		fox2.close();
		cout << "surfaceRel2OilX!" << endl;
		ifstream foy2("data/SurfRel2OilY.dat");
		try
		{
			if (!foy2) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel2OilY.dat" << endl;
		}
		getline(foy2, field_comment);
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 2); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
				{
					foy2 >> buf;
					if (int((k - 1) / gs.kk) == MPI_rank)
					{
						surfaceRel2Oil.y[index(i, j, ((k - 1) % gs.kk + 1), gs)] = buf;
					}
				}
			}
		}
		foy2.close();
		cout << "SurfRel2OilY done!" << endl;
		ifstream foz2("data/SurfRel2OilZ.dat");
		try
		{
			if (!foz2) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file SurfRel2OilZ.dat" << endl;
		}
		getline(foz2, field_comment);
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 2); k++)
				{
					foz2 >> buf;
					bufArray[index(i, j, k, gs)] = buf;
				}
			}
		}
		foz2.close();
		cout << "SurfRel2OilZ done!" << endl;
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk + 2); k++)
				{
					surfaceRel2Oil.z[index(i, j, k, gs)] = bufArray[index(i, j, k + MPI_rank*gs.kk, gs)];
				}
			}
		}
		delete[] bufArray;
	}
	else
	{
		for (int k = 0; k < (gs.kk + 2); k++)
		{
			for (int j = 0; j < (gs.jk + 2); j++)
			{
				for (int i = 0; i < (gs.ik + 2); i++)
				{
					surfaceRel1Oil.x[index(i, j, k, gs)] = r1Oil;
					surfaceRel1Oil.y[index(i, j, k, gs)] = r1Oil;
					surfaceRel1Oil.z[index(i, j, k, gs)] = r1Oil;
					surfaceRel2Oil.x[index(i, j, k, gs)] = r2Oil;
					surfaceRel2Oil.y[index(i, j, k, gs)] = r2Oil;
					surfaceRel2Oil.z[index(i, j, k, gs)] = r2Oil;
				}
			}
		}
	}
}

void NmrRelaxationSolve::ReadVelocity()
{
	double velocityMax = 0.0;
	double velocityMin = 0.0;
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				velocity.x[index(i, j, k, gs)] = 0.0;
				velocity.y[index(i, j, k, gs)] = 0.0;
				velocity.z[index(i, j, k, gs)] = 0.0;
			}
		}
	}
	dTimeVel = 1.0;
	if (nVelocity == 1)
	{
		double buf;

		std::string field_comment;
		ifstream f1("data/Vx.dat");
		try
		{
			if (!f1) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file Vx.dat" << endl;
		}
		getline(f1, field_comment);
		for (int i = 1; i < (gs.ik + 2); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
				{
					f1 >> buf;
					if (int((k - 1) / gs.kk) == MPI_rank)
					{
						velocity.x[index(i, j, ((k-1) % gs.kk + 1), gs)] = velocityMultiplicator*buf;
					}
					if (velocityMax < buf) velocityMax = buf;
					if (velocityMin > buf) velocityMin = buf;
				}
			}
		}
		f1.close();
		cout << "Vx_done!" << endl;
		ifstream f2("data/Vy.dat");
		try
		{
			if (!f2) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file Vy.dat" << endl;
		}
		getline(f2, field_comment);
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 2); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 1); k++)
				{
					f2 >> buf;
					if (int((k - 1) / gs.kk) == MPI_rank)
					{
						velocity.y[index(i, j, ((k-1) % gs.kk + 1), gs)] = velocityMultiplicator*buf;
					}
					if (velocityMax < buf) velocityMax = buf;
					if (velocityMin > buf) velocityMin = buf;
				}
			}
		}
		f2.close();
		cout << "Vy_done!" << endl;
		//
		ifstream f3("data/Vz.dat");
		try
		{
			if (!f3) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file Vz.dat" << endl;
		}
		getline(f3, field_comment);
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk + 2); k++)
				{
					f3 >> buf;
					velocity.z[index(i, j, k, gs)] = velocityMultiplicator*buf;
					if (velocityMax < buf) velocityMax = buf;
					if (velocityMin > buf) velocityMin = buf;
				}
			}
		}
		f3.close();
		cout << "Vz_done!" << endl;
		//
		/*ifstream f3("data/Vz.dat");
		try
		{
			if (!f3) throw 1;
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " No file Vz.dat" << endl;
		}
		getline(f3, field_comment);
		double *bufArray;
		bufArray = new double[gs.num*(gs.kk*MPI_size + 2) / (gs.kk + 2)];
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk*MPI_size + 2); k++)
				{
					f3 >> buf;
					if (velocityMax < buf) velocityMax = buf;
					if (velocityMin > buf) velocityMin = buf;
					bufArray[index(i, j, k, gs)] = velocityMultiplicator*buf;
					//cout << "i=" << i << " j=" << j << " k=" << k << endl;
				}
			}
		}
		f3.close();
		cout << "Vz_done!" << endl;
		for (int i = 1; i < (gs.ik + 1); i++)
		{
			for (int j = 1; j < (gs.jk + 1); j++)
			{
				for (int k = 1; k < (gs.kk + 2); k++)
				{
					velocity.z[index(i, j, k, gs)] = bufArray[index(i, j, k+MPI_rank*gs.kk, gs)];
					//cout << "i=" << i << " j=" << j << " k=" << endl;
				}
			}
		}
		
		delete[] bufArray;*/
		cout << "velocityMax=" << scientific << velocityMax << endl;
		cout << "velocityMin=" << scientific << velocityMin << endl;
		if (velocityMax > abs(velocityMin)) velocityMax = velocityMax*velocityMultiplicator;
		else velocityMax = abs(velocityMin)*velocityMultiplicator;
		if ((dx <= dy) && (dx <= dz)) dTimeVel = cfl*dx / (velocityMax + small);
		if ((dy <= dy) && (dy <= dz)) dTimeVel = cfl*dy / (velocityMax + small);
		if ((dz <= dy) && (dz <= dz)) dTimeVel = cfl*dz / (velocityMax + small);

	}
	cout << "dTimeVel=" << scientific << dTimeVel << " rank = " << MPI_rank << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	int area;
	double velocityTotal;
	velocityTotal = 0.0;
	area = 0;
	
	if (nVelocity == 1)
	{
		if (nDirectionField == 1)
		{
			for (int k = 1; k < (gs.kk + 1); k++)
			{
				for (int j = 1; j < (gs.jk + 1); j++)
				{
					if (actCell[index(1, j, k, gs)] == '1')
					{
						velocityTotal += velocity.x[index(1, j, k, gs)];
						area += 1;
					}
				}
			}
			velocityAverage = velocityTotal * area / (gs.kk * gs.jk);
			cout << "Average velocity = " << scientific << velocityAverage << endl;
		}
		if (nDirectionField == 2)
		{
			for (int k = 1; k < (gs.kk + 1); k++)
			{
				for (int i = 1; i < (gs.ik + 1); i++)
				{
					if (actCell[index(i, 1, k, gs)] == '1')
					{
						velocityTotal += velocity.y[index(i, 1, k, gs)];
						area += 1;
					}
				}
			}
			velocityAverage = velocityTotal * area / (gs.kk * gs.ik);
			cout << "Average velocity = " << scientific << velocityAverage << endl;
		}
		if (nDirectionField == 3)
		{
			for (int i = 1; i < (gs.ik + 1); i++)
			{
				for (int j = 1; j < (gs.jk + 1); j++)
				{
					if (actCell[index(i, j, 1, gs)] == '1')
					{
						velocityTotal += velocity.z[index(i, j, 1, gs)];
						area += 1;
					}
				}
			}
			velocityAverage = velocityTotal * area / (gs.ik * gs.jk);
			cout << "Average velocity = " << scientific << velocityAverage << endl;
		}
		try
		{
			if (nDirectionField != 1 && nDirectionField != 2 && nDirectionField != 3) throw 2;
			
		}
		catch (int a)
		{
			cout << "Caught exception number:  " << a << " Can work out the direction of the flow. nDirectField = " << nDirectionField << endl;
		}
	}
	else
	{
		velocityAverage = 0.0;  
	}
	
}

void NmrRelaxationSolve::BoundaryConditionsForVectorField(VectorField *field)
{
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			field->x[index(0, j, k, gs)] = field->x[index(1, j, k, gs)];
			field->y[index(0, j, k, gs)] = field->y[index(1, j, k, gs)];
			field->z[index(0, j, k, gs)] = field->z[index(1, j, k, gs)];
			field->x[index(gs.ik + 1, j, k, gs)] = field->x[index(gs.ik, j, k, gs)];
			field->y[index(gs.ik + 1, j, k, gs)] = field->y[index(gs.ik, j, k, gs)];
			field->z[index(gs.ik + 1, j, k, gs)] = field->z[index(gs.ik, j, k, gs)];
		}
	}
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int i = 0; i < (gs.ik + 2); i++)
		{
			field->x[index(i, 0, k, gs)] = field->x[index(i, 1, k, gs)];
			field->y[index(i, 0, k, gs)] = field->y[index(i, 1, k, gs)];
			field->z[index(i, 0, k, gs)] = field->z[index(i, 1, k, gs)];
			field->x[index(i, gs.jk + 1, k, gs)] = field->x[index(i, gs.jk, k, gs)];
			field->y[index(i, gs.jk + 1, k, gs)] = field->y[index(i, gs.jk, k, gs)];
			field->z[index(i, gs.jk + 1, k, gs)] = field->z[index(i, gs.jk, k, gs)];
		}
	}
	//MPI boundary conditions
	MPI_Status status;
	for (int j = 0; j < (gs.jk + 2); j++)
	{
		for (int i = 0; i < (gs.ik + 2); i++)
		{
			MPI_Sendrecv(&field->x[index(i, j, gs.kk, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
				&field->x[index(i, j, 0, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
				MPI_COMM_WORLD, &status);
			MPI_Sendrecv(&field->x[index(i, j, 1, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
				&field->x[index(i, j, gs.kk + 1, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
				MPI_COMM_WORLD, &status);
		}
	}
	for (int j = 0; j < (gs.jk + 2); j++)
	{
		for (int i = 0; i < (gs.ik + 2); i++)
		{
			MPI_Sendrecv(&field->y[index(i, j, gs.kk, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
				&field->y[index(i, j, 0, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
				MPI_COMM_WORLD, &status);
			MPI_Sendrecv(&field->y[index(i, j, 1, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
				&field->y[index(i, j, gs.kk + 1, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
				MPI_COMM_WORLD, &status);
		}
	}
	for (int j = 0; j < (gs.jk + 2); j++)
	{
		for (int i = 0; i < (gs.ik + 2); i++)
		{
			MPI_Sendrecv(&field->z[index(i, j, gs.kk, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
				&field->z[index(i, j, 0, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
				MPI_COMM_WORLD, &status);
			MPI_Sendrecv(&field->z[index(i, j, 1, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
				&field->z[index(i, j, gs.kk + 1, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
				MPI_COMM_WORLD, &status);
		}
	}
	//end of MPI boundary conditions. Corrections within 0 & size-1 ranks
	if (MPI_rank == 0)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				field->x[index(i, j, 0, gs)] = field->x[index(i, j, 1, gs)];
				field->y[index(i, j, 0, gs)] = field->y[index(i, j, 1, gs)];
				field->z[index(i, j, 0, gs)] = field->z[index(i, j, 1, gs)];
			}
		}
	}
	if (MPI_rank == (MPI_size-1))
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				field->x[index(i, j, gs.kk + 1, gs)] = field->x[index(i, j, gs.kk, gs)];
				field->y[index(i, j, gs.kk + 1, gs)] = field->y[index(i, j, gs.kk, gs)];
				field->z[index(i, j, gs.kk + 1, gs)] = field->z[index(i, j, gs.kk, gs)];
			}
		}
	}

}

void NmrRelaxationSolve::FillInDiffusinAndTime()
{
	int km, kp, jm, jp, im, ip;
	double z1, z2, c1, c2, ct1, ct2, cd, dMax=0.0, tMin=100.0, dM, dT;
	for (int k = 1; k < (gs.kk + 1); k++)
	{
		km = k - 1;
		kp = k + 1;
		for (int j = 1; j < (gs.jk + 1); j++)
		{
			jm = j - 1;
			jp = j + 1;
			for (int i = 1; i < (gs.ik + 1); i++)
			{
				im = i - 1;
				ip = i + 1;
				if (actCell[index(i, j, k, gs)] == '0')
				{
					diffusion[index(i, j, k, gs)] = small;
					relTime.t1[index(i, j, k, gs)] = big;
					relTime.t2[index(i, j, k, gs)] = big;
				}
				if (actCell[index(i, j, k, gs)] == '1')
				{
					if (nComponents == 2)
					{
						z1 = sqrt((concentration[0][index(i, j, k, gs)] - an1) * (concentration[0][index(i, j, k, gs)] - an1) + (concentration[1][index(i, j, k, gs)] - an2) * (concentration[1][index(i, j, k, gs)] - an2));
						z2 = sqrt((concentration[0][index(i, j, k, gs)] - bn1) * (concentration[0][index(i, j, k, gs)] - bn1) + (concentration[1][index(i, j, k, gs)] - bn2) * (concentration[1][index(i, j, k, gs)] - bn2));
						c1 = z2 / (z1 + z2);
						c2 = z1 / (z1 + z2);
					}
					if (nComponents == 1)
					{
						c1 = concentration[0][index(i, j, k, gs)];
						c2 = concentration[1][index(i, j, k, gs)];
					}
					cd = (c1*dWater + c2*dOil);
					diffusion[index(i, j, k, gs)] = cd*pow((1.0-2.0*c1*c2/(c1*c1+c2*c2)),10.0);
					if (nDiffusion == 0) diffusion[index(i, j, k, gs)] = small;
					ct1 = c1/t1Water+c2/t1Oil;
					ct2 = c1/t2Water+c2/t2Oil;
					//������� �� ������� ���� ������
					if (actCell[index(im, j, k, gs)] == '0')
					{
						ct1 += (c1*surfaceRel1Water.x[index(i, j, k, gs)] + c2*surfaceRel1Oil.x[index(i, j, k, gs)]) / dx;
						ct2 += (c1*surfaceRel2Water.x[index(i, j, k, gs)] + c2*surfaceRel2Oil.x[index(i, j, k, gs)]) / dx;
						//ct1 += (c1*r1Water + c2*r1Oil) / dx;
						//ct2 += (c1*r2Water + c2*r2Oil) / dx;
					}
					if (actCell[index(ip, j, k, gs)] == '0')
					{
						ct1 += (c1*surfaceRel1Water.x[index(ip, j, k, gs)] + c2*surfaceRel1Oil.x[index(ip, j, k, gs)]) / dx;
						ct2 += (c1*surfaceRel2Water.x[index(ip, j, k, gs)] + c2*surfaceRel2Oil.x[index(ip, j, k, gs)]) / dx;
						//ct1 += (c1*r1Water + c2*r1Oil) / dx;
						//ct2 += (c1*r2Water + c2*r2Oil) / dx;
					}
					if (actCell[index(i, jm, k, gs)] == '0')
					{
						ct1 += (c1*surfaceRel1Water.y[index(i, j, k, gs)] + c2*surfaceRel1Oil.y[index(i, j, k, gs)]) / dy;
						ct2 += (c1*surfaceRel2Water.y[index(i, j, k, gs)] + c2*surfaceRel2Oil.y[index(i, j, k, gs)]) / dy;
						//ct1 += (c1*r1Water + c2*r1Oil) / dy;
						//ct2 += (c1*r2Water + c2*r2Oil) / dy;
					}
					if (actCell[index(i, jp, k, gs)] == '0')
					{
						ct1 += (c1*surfaceRel1Water.y[index(i, jp, k, gs)] + c2*surfaceRel1Oil.y[index(i, jp, k, gs)]) / dy;
						ct2 += (c1*surfaceRel2Water.y[index(i, jp, k, gs)] + c2*surfaceRel2Oil.y[index(i, jp, k, gs)]) / dy;
						//ct1 += (c1*r1Water + c2*r1Oil) / dy;
						//ct2 += (c1*r2Water + c2*r2Oil) / dy;
					}
					if (actCell[index(i, j, km, gs)] == '0')
					{
						ct1 += (c1*surfaceRel1Water.z[index(i, j, k, gs)] + c2*surfaceRel1Oil.z[index(i, j, k, gs)]) / dz;
						ct2 += (c1*surfaceRel2Water.z[index(i, j, k, gs)] + c2*surfaceRel2Oil.z[index(i, j, k, gs)]) / dz;
						//ct1 += (c1*r1Water + c2*r1Oil) / dz;
						//ct2 += (c1*r2Water + c2*r2Oil) / dz;
					}
					if (actCell[index(i, j, kp, gs)] == '0')
					{
						ct1 += (c1*surfaceRel1Water.z[index(i, j, kp, gs)] + c2*surfaceRel1Oil.z[index(i, j, kp, gs)]) / dz;
						ct2 += (c1*surfaceRel2Water.z[index(i, j, kp, gs)] + c2*surfaceRel2Oil.z[index(i, j, kp, gs)]) / dz;
						//ct1 += (c1*r1Water + c2*r1Oil) / dz;
						//ct2 += (c1*r2Water + c2*r2Oil) / dz;
					}
					relTime.t1[index(i, j, k, gs)] = 1.0 / ct1;
					relTime.t2[index(i, j, k, gs)] = 1.0 / ct2;
					if (dMax <= diffusion[index(i, j, k, gs)]) dMax = diffusion[index(i, j, k, gs)];
					if (tMin >= relTime.t1[index(i, j, k, gs)]) tMin = relTime.t1[index(i, j, k, gs)];
					if (tMin >= relTime.t2[index(i, j, k, gs)]) tMin = relTime.t2[index(i, j, k, gs)];
				}
			}
		}
	}
	MPI_Allreduce(&dMax, &dM, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tMin, &dT, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (nDiffusion == 1)
	{
		if ((dx <= dy) && (dx <= dz)) dTimeDiff = 2.0*cfl*dx*dx / dM;
		if ((dy <= dy) && (dy <= dz)) dTimeDiff = 2.0*cfl*dy*dy / dM;
		if ((dz <= dy) && (dz <= dz)) dTimeDiff = 2.0*cfl*dz*dz / dM;
	}
	else
	{
		dTimeDiff = 1.0;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	cout << "dTimeDiff=" << scientific <<dTimeDiff << " rank=" << MPI_rank << endl;
	dTimeRel = cfl1*dT;
	cout << "dTimeRel=" << scientific << dTimeRel << " rank=" << MPI_rank << endl;

	/*std::ofstream time1("Time_1.csv");
	for (int k = 1; k < (gs.kk + 1); k = k++)
	{
		for (int j = 1; j < (gs.jk + 1); j = j++)
		{
			for (int i = 1; i < (gs.ik + 1); i = i++)
			{
				time1 << scientific << relTime.t1[index(i, j, k, gs)] << ";";
			}
			time1 << endl;
		}
	}
	time1.close();
	std::ofstream time2("Time_2.csv");
	for (int k = 1; k < (gs.kk + 1); k = k++)
	{
		for (int j = 1; j < (gs.jk + 1); j = j++)
		{
			for (int i = 1; i < (gs.ik + 1); i = i++)
			{
				time2 << scientific << relTime.t2[index(i, j, k, gs)] << ";";
			}
			time2 << endl;
		}
	}
	time2.close();*/

}

void NmrRelaxationSolve::BoundaryConditionsForDiffusion()
{
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			diffusion[index(0, j, k, gs)] = diffusion[index(1, j, k, gs)];
			diffusion[index(gs.ik + 1, j, k, gs)] = diffusion[index(gs.ik, j, k, gs)];
		}
	}
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int i = 0; i < (gs.ik + 2); i++)
		{
			diffusion[index(i, 0, k, gs)] = diffusion[index(i, 1, k, gs)];
			diffusion[index(i, gs.jk + 1, k, gs)] = diffusion[index(i, gs.jk, k, gs)];
		}
	}
	//MPI boundary conditions
	MPI_Status status;
	MPI_Sendrecv(&diffusion[index(0, 0, gs.kk, gs)], (gs.ik + 2)*(gs.jk + 2), MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
		&diffusion[index(0, 0, 0, gs)], (gs.ik + 2)*(gs.jk + 2), MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
		MPI_COMM_WORLD, &status);
	MPI_Sendrecv(&diffusion[index(0, 0, 1, gs)], (gs.ik + 2)*(gs.jk + 2), MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
		&diffusion[index(0, 0, gs.kk + 1, gs)], (gs.ik + 2)*(gs.jk + 2), MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
		MPI_COMM_WORLD, &status);
	//end of MPI boundary conditions. Corrections within 0 & size-1 ranks
	if (MPI_rank == 0)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				diffusion[index(i, j, 0, gs)] = diffusion[index(i, j, 1, gs)];
			}
		}
	}
	if (MPI_rank == (MPI_size - 1) )
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				diffusion[index(i, j, gs.kk + 1, gs)] = diffusion[index(i, j, gs.kk, gs)];
			}
		}
	}
}

void NmrRelaxationSolve::FillInMainField()
{
	double gradAmpl;
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				mainField.x[index(i, j, k, gs)] = 0.0;
				mainField.y[index(i, j, k, gs)] = 0.0;
				mainField.z[index(i, j, k, gs)] = 0.0;
				if (nDirectionField == 1) mainField.x[index(i, j, k, gs)] = bsx;
				if (nDirectionField == 2) mainField.y[index(i, j, k, gs)] = bsy;
				if (nDirectionField == 3) mainField.z[index(i, j, k, gs)] = bsz;
				if (nDirectionField == 0)
				{
					mainField.x[index(i, j, k, gs)] = bsx;
					mainField.y[index(i, j, k, gs)] = bsy;
					mainField.z[index(i, j, k, gs)] = bsz;
				}
				//printf("Field = %f %f %f\n", mainField.x[index(i, j, k, gs)], mainField.y[index(i, j, k, gs)], mainField.z[index(i, j, k, gs)]);
				noiseField.x[index(i, j, k, gs)] = 0.0;
				noiseField.y[index(i, j, k, gs)] = 0.0;
				noiseField.z[index(i, j, k, gs)] = 0.0;
				//if (flag == 1) angle = grad*g*((i - 0.5)*dx*nx + (j - 0.5)*dy*ny + (k - 0.5)*dz*nz);
				noiseField.x[index(i, j, k, gs)] = constGradX*dx*(i - 0.5);
				noiseField.y[index(i, j, k, gs)] = constGradY*dy*(j - 0.5);
				noiseField.z[index(i, j, k, gs)] = constGradZ*dz*(k - 0.5);
				if (nDirectionField != 0 && noiseAmpl != 0.0)
				{
					if (nDirectionField == 1) noiseField.x[index(i, j, k, gs)] = noiseAmpl*(rand() % 2000 + 1 - 1000) / 1000;
					if (nDirectionField == 2) noiseField.y[index(i, j, k, gs)] = noiseAmpl*(rand() % 2000 + 1 - 1000) / 1000;
					if (nDirectionField == 3) noiseField.z[index(i, j, k, gs)] = noiseAmpl*(rand() % 2000 + 1 - 1000) / 1000;
				}

			}
		}
	}
	if (constGradX != 0) gradAmpl = abs(constGradX*dx*gs.ik); 
	if (constGradY != 0) gradAmpl = abs(constGradY*dy*gs.jk);
	if (constGradZ != 0) gradAmpl = abs(constGradZ*dz*gs.kk);
	if (constGradX == 0 && constGradY == 0 && constGradZ == 0) gradAmpl = small;
	dTimeNoise = cfl1 / (g * noiseAmpl);
	dTimeGrad = cfl1 / (g * gradAmpl);
}

void NmrRelaxationSolve::PrintActiveCells()
{
	for (int k = 0; k < (gs.kk + 2); k=k+kPrint)
	{
		cout << "Layer = " << k << endl;
		for (int j = 0; j < (gs.jk + 2); j=j+jPrint)
		{
			for (int i = 0; i < (gs.ik + 2); i=i+iPrint)
			{
				cout << "[" << i << "," << j << "," << k << "]=" << actCell[index(i, j, k, gs)] << "	";
				printf("[%d,%d,%d]=%c ", i, j, k, actCell[index(i, j, k, gs)]);
			}
			cout << endl;
		}
	}
	cout << endl;
}

void NmrRelaxationSolve::PrintConcentration()
{
	for (int nPhase = 0; nPhase < 2; nPhase++)
	{
		cout << "Phase num" << nPhase + 1 << endl;
		for (int k = 0; k < (gs.kk + 2); k = k + kPrint)
		{
			cout << "Layer = " << k << endl;
			for (int j = 0; j < (gs.jk + 2); j = j + jPrint)
			{
				for (int i = 0; i < (gs.ik + 2); i = i + iPrint)
				{
					cout << "[" << i << "," << j << "," << k << "]=" << scientific << concentration[nPhase][index(i, j, k, gs)] << "	";
				}
				cout << endl;
			}
		}
	}
	cout << endl;
}

void NmrRelaxationSolve::PrintVectorField(int flag, VectorField *field)
{
	double x, y, z;
	if (flag == 0)
	{
		cout << "X component" << endl;
		for (int k = 0; k < (gs.kk + 2); k = k + kPrint)
		{
			cout << "Layer = " << k << endl;
			for (int j = 0; j < (gs.jk + 2); j = j + jPrint)
			{
				for (int i = 0; i < (gs.ik + 2); i = i + iPrint)
				{
					x = field->x[index(i, j, k, gs)];
					cout << "[" << i << "," << j << "," << k << "]=" << scientific << x << "	";
				}
				cout << endl;
			}
		}
		cout << "Y component" << endl;
		for (int k = 0; k < (gs.kk + 2); k = k + kPrint)
		{
			cout << "Layer = " << k << endl;
			for (int j = 0; j < (gs.jk + 2); j = j + jPrint)
			{
				for (int i = 0; i < (gs.ik + 2); i = i + iPrint)
				{

					y = field->y[index(i, j, k, gs)];
					cout << "[" << i << "," << j << "," << k << "]=" << scientific << y << "	";
				}
				cout << endl;
			}
		}
		cout << "Z component" << endl;
		for (int k = 0; k < (gs.kk + 2); k = k + kPrint)
		{
			cout << "Layer = " << k << endl;
			for (int j = 0; j < (gs.jk + 2); j = j + jPrint)
			{
				for (int i = 0; i < (gs.ik + 2); i = i + iPrint)
				{
					z = field->z[index(i, j, k, gs)];
					cout << "[" << i << "," << j << "," << k << "]=" << scientific << z << "	";
				}
				cout << endl;
			}
		}
		cout << endl;
	}
	if (flag == 1)
	{
		cout << "X component" << endl;
		for (int k = 1; k < (gs.kk + 1); k = k + kPrint)
		{
			cout << "Layer = " << k << endl;
			for (int j = 1; j < (gs.jk + 1); j = j + jPrint)
			{
				for (int i = 1; i < (gs.ik + 1); i = i + iPrint)
				{
					x = field->x[index(i, j, k, gs)];
					cout << "[" << i << "," << j << "," << k << "]=" << scientific << x << "	";
				}
				cout << endl;
			}
		}
		cout << "Y component" << endl;
		for (int k = 1; k < (gs.kk + 1); k = k + kPrint)
		{
			cout << "Layer = " << k << endl;
			for (int j = 1; j < (gs.jk + 1); j = j + jPrint)
			{
				for (int i = 1; i < (gs.ik + 1); i = i + iPrint)
				{

					y = field->y[index(i, j, k, gs)];
					cout << "[" << i << "," << j << "," << k << "]=" << scientific << y << "	";
				}
				cout << endl;
			}
		}
		cout << "Z component" << endl;
		for (int k = 1; k < (gs.kk + 1); k = k + kPrint)
		{
			cout << "Layer = " << k << endl;
			for (int j = 1; j < (gs.jk + 1); j = j + jPrint)
			{
				for (int i = 1; i < (gs.ik + 1); i = i + iPrint)
				{
					z = field->z[index(i, j, k, gs)];
					cout << "[" << i << "," << j << "," << k << "]=" << scientific << z << "	";
				}
				cout << endl;
			}
		}
		cout << endl;
	}
}

void NmrRelaxationSolve::WriteVectorField(int flag, VectorField *field, string fileName)
{
	std::ofstream test(fileName.c_str());
	double x, y, z;
	if (flag == 0)
	{
		test << "X component" << endl;
		for (int k = 0; k < (gs.kk + 2); k = k + kPrint)
		{
			test << "Layer = " << k << endl;
			for (int j = 0; j < (gs.jk + 2); j = j + jPrint)
			{
				for (int i = 0; i < (gs.ik + 2); i = i + iPrint)
				{
					x = field->x[index(i, j, k, gs)];
					test << "[" << i << "," << j << "," << k << "]=" << scientific << x << "	";
				}
				test << endl;
			}
		}
		test << "Y component" << endl;
		for (int k = 0; k < (gs.kk + 2); k = k + kPrint)
		{
			test << "Layer = " << k << endl;
			for (int j = 0; j < (gs.jk + 2); j = j + jPrint)
			{
				for (int i = 0; i < (gs.ik + 2); i = i + iPrint)
				{

					y = field->y[index(i, j, k, gs)];
					test << "[" << i << "," << j << "," << k << "]=" << scientific << y << "	";
				}
				test << endl;
			}
		}
		test << "Z component" << endl;
		for (int k = 0; k < (gs.kk + 2); k = k + kPrint)
		{
			test << "Layer = " << k << endl;
			for (int j = 0; j < (gs.jk + 2); j = j + jPrint)
			{
				for (int i = 0; i < (gs.ik + 2); i = i + iPrint)
				{
					z = field->z[index(i, j, k, gs)];
					test << "[" << i << "," << j << "," << k << "]=" << scientific << z << "	";
				}
				test << endl;
			}
		}
		test << endl;
	}
	if (flag == 1)
	{
		test << "X component" << endl;
		for (int k = 1; k < (gs.kk + 1); k = k + kPrint)
		{
			test << "Layer = " << k << endl;
			for (int j = 1; j < (gs.jk + 1); j = j + jPrint)
			{
				for (int i = 1; i < (gs.ik + 1); i = i + iPrint)
				{
					x = field->x[index(i, j, k, gs)];
					test << "[" << i << "," << j << "," << k << "]=" << scientific << x << "	";
				}
				test << endl;
			}
		}
		test << "Y component" << endl;
		for (int k = 1; k < (gs.kk + 1); k = k + kPrint)
		{
			test << "Layer = " << k << endl;
			for (int j = 1; j < (gs.jk + 1); j = j + jPrint)
			{
				for (int i = 1; i < (gs.ik + 1); i = i + iPrint)
				{

					y = field->y[index(i, j, k, gs)];
					test << "[" << i << "," << j << "," << k << "]=" << scientific << y << "	";
				}
				test << endl;
			}
		}
		test << "Z component" << endl;
		for (int k = 1; k < (gs.kk + 1); k = k + kPrint)
		{
			test << "Layer = " << k << endl;
			for (int j = 1; j < (gs.jk + 1); j = j + jPrint)
			{
				for (int i = 1; i < (gs.ik + 1); i = i + iPrint)
				{
					z = field->z[index(i, j, k, gs)];
					test << "[" << i << "," << j << "," << k << "]=" << scientific << z << "	";
				}
				test << endl;
			}
		}
		test << endl;
	}
}

void NmrRelaxationSolve::WriteVectorFieldSeparated(VectorField *field, string fileNamePattern)
{
	if (nComponents == 1)
	{
		double x, y, z, mod;
		std::ofstream testx((fileNamePattern + string("_X.csv")).c_str());
		for (int k = 1; k < (gs.kk + 1); k = k++)
		{
			for (int j = 1; j < (gs.jk + 1); j = j++)
			{
				for (int i = 1; i < (gs.ik + 1); i = i++)
				{
					if (actCell[index(i, j, k, gs)] == '1') x = field->x[index(i, j, k, gs)];
					else x = 0.0;
					if (i != (gs.ik + 0)) testx << scientific << x << ";";
					else testx << scientific << x;
				}
				testx << endl;
			}
		}
		testx.close();
		std::ofstream testy((fileNamePattern + string("_Y.csv")).c_str());
		for (int k = 1; k < (gs.kk + 1); k = k++)
		{
			for (int j = 1; j < (gs.jk + 1); j = j++)
			{
				for (int i = 1; i < (gs.ik + 1); i = i++)
				{
					if (actCell[index(i, j, k, gs)] == '1') y = field->y[index(i, j, k, gs)];
					else y = 0.0;
					if (i != (gs.ik + 0)) testy << scientific << y << ";";
					else testy << scientific << y;
				}
				testy << endl;
			}
		}
		testy.close();
		std::ofstream testz((fileNamePattern + string("_Z.csv")).c_str());
		for (int k = 1; k < (gs.kk + 1); k = k++)
		{
			for (int j = 1; j < (gs.jk + 1); j = j++)
			{
				for (int i = 1; i < (gs.ik + 1); i = i++)
				{
					if (actCell[index(i, j, k, gs)] == '1') z = field->z[index(i, j, k, gs)];
					else z = 0.0;
					if (i != (gs.ik + 0)) testz << scientific << z << ";";
					else testz << scientific << z;
				}
				testz << endl;
			}
		}
		testz.close();
		/*std::ofstream testmod((fileNamePattern + string("Abs.csv")).c_str());
		for (int k = 1; k < (gs.kk + 1); k = k++)
		{
			for (int j = 1; j < (gs.jk + 1); j = j++)
			{
				for (int i = 1; i < (gs.ik + 1); i = i++)
				{
					if (actCell[index(i, j, k, gs)] == '1') mod = sqrt(field->x[index(i, j, k, gs)] * field->x[index(i, j, k, gs)] + field->y[index(i, j, k, gs)] * field->y[index(i, j, k, gs)] + field->z[index(i, j, k, gs)] * field->z[index(i, j, k, gs)]);
					else mod = 0.0;
					testmod << scientific << mod << ";";
				}
				testmod << endl;
			}
		}
		testmod.close();*/
	}
	if (nComponents == 2)
	{
		double z1, z2, c1, c2, xw, xo, yw, yo, zw, zo;
		std::ofstream testWx((fileNamePattern + string("Xwat.csv")).c_str());
		std::ofstream testOx((fileNamePattern + string("Xoil.csv")).c_str());
		for (int k = 1; k < (gs.kk + 1); k = k++)
		{
			for (int j = 1; j < (gs.jk + 1); j = j++)
			{
				for (int i = 1; i < (gs.ik + 1); i = i++)
				{
					if (actCell[index(i, j, k, gs)] == '1')
					{
						z1 = sqrt((concentration[0][index(i, j, k, gs)] - an1) * (concentration[0][index(i, j, k, gs)] - an1) + (concentration[1][index(i, j, k, gs)] - an2) * (concentration[1][index(i, j, k, gs)] - an2));
						z2 = sqrt((concentration[0][index(i, j, k, gs)] - bn1) * (concentration[0][index(i, j, k, gs)] - bn1) + (concentration[1][index(i, j, k, gs)] - bn2) * (concentration[1][index(i, j, k, gs)] - bn2));
						c1 = z2 / (z1 + z2);
						c2 = z1 / (z1 + z2);
						xw = field->x[index(i, j, k, gs)] * c1;
						xo = field->x[index(i, j, k, gs)] * c2;
					}
					else
					{
						xw = 0.0;
						xo = 0.0;
					}
					testWx << scientific << xw << ";";
					testOx << scientific << xo << ";";
				}
				testWx << endl;
				testOx << endl;
			}
		}
		testWx.close();
		testOx.close();
		std::ofstream testWy((fileNamePattern + string("Ywat.csv")).c_str());
		std::ofstream testOy((fileNamePattern + string("Yoil.csv")).c_str());
		for (int k = 1; k < (gs.kk + 1); k = k++)
		{
			for (int j = 1; j < (gs.jk + 1); j = j++)
			{
				for (int i = 1; i < (gs.ik + 1); i = i++)
				{
					if (actCell[index(i, j, k, gs)] == '1')
					{
						z1 = sqrt((concentration[0][index(i, j, k, gs)] - an1) * (concentration[0][index(i, j, k, gs)] - an1) + (concentration[1][index(i, j, k, gs)] - an2) * (concentration[1][index(i, j, k, gs)] - an2));
						z2 = sqrt((concentration[0][index(i, j, k, gs)] - bn1) * (concentration[0][index(i, j, k, gs)] - bn1) + (concentration[1][index(i, j, k, gs)] - bn2) * (concentration[1][index(i, j, k, gs)] - bn2));
						c1 = z2 / (z1 + z2);
						c2 = z1 / (z1 + z2);
						yw = field->y[index(i, j, k, gs)] * c1;
						yo = field->y[index(i, j, k, gs)] * c2;
					}
					else
					{
						yw = 0.0;
						yo = 0.0;
					}
					testWy << scientific << yw << ";";
					testOy << scientific << yo << ";";
				}
				testWy << endl;
				testOy << endl;
			}
		}
		testWy.close();
		testOy.close();
		std::ofstream testWz((fileNamePattern + string("Zwat.csv")).c_str());
		std::ofstream testOz((fileNamePattern + string("Zoil.csv")).c_str());
		for (int k = 1; k < (gs.kk + 1); k = k++)
		{
			for (int j = 1; j < (gs.jk + 1); j = j++)
			{
				for (int i = 1; i < (gs.ik + 1); i = i++)
				{
					if (actCell[index(i, j, k, gs)] == '1')
					{
						z1 = sqrt((concentration[0][index(i, j, k, gs)] - an1) * (concentration[0][index(i, j, k, gs)] - an1) + (concentration[1][index(i, j, k, gs)] - an2) * (concentration[1][index(i, j, k, gs)] - an2));
						z2 = sqrt((concentration[0][index(i, j, k, gs)] - bn1) * (concentration[0][index(i, j, k, gs)] - bn1) + (concentration[1][index(i, j, k, gs)] - bn2) * (concentration[1][index(i, j, k, gs)] - bn2));
						c1 = z2 / (z1 + z2);
						c2 = z1 / (z1 + z2);
						zw = field->z[index(i, j, k, gs)] * c1;
						zo = field->z[index(i, j, k, gs)] * c2;
					}
					else
					{
						zw = 0.0;
						zo = 0.0;
					}
					testWz << scientific << zw << ";";
					testOz << scientific << zo << ";";
				}
				testWz << endl;
				testOz << endl;
			}
		}
		testWz.close();
		testOz.close();
	}
}

void NmrRelaxationSolve::PrintDiffusionAndRelTime()
{
	cout << "D" << endl;
	for (int k = 0; k < (gs.kk + 2); k = k + kPrint)
	{
		cout << "Layer = " << k << endl;
		for (int j = 0; j < (gs.jk + 2); j = j + jPrint)
		{
			for (int i = 0; i < (gs.ik + 2); i = i + iPrint)
			{
				cout << "[" << i << "," << j << "," << k << "]=" << scientific << diffusion[index(i, j, k, gs)] << "	";
			}
			cout << endl;
		}
	}
	cout << "T1" << endl;
	for (int k = 1; k < (gs.kk + 1); k = k + kPrint)
	{
		cout << "Layer = " << k << endl;
		for (int j = 1; j < (gs.jk + 1); j = j + jPrint)
		{
			for (int i = 1; i < (gs.ik + 1); i = i + iPrint)
			{
				cout << "[" << i << "," << j << "," << k << "]=" << scientific << relTime.t1[index(i, j, k, gs)] << "	";
			}
			cout << endl;
		}
	}
	cout << "T2" << endl;
	for (int k = 1; k < (gs.kk + 1); k = k + kPrint)
	{
		cout << "Layer = " << k << endl;
		for (int j = 1; j < (gs.jk + 1); j = j + jPrint)
		{
			for (int i = 1; i < (gs.ik + 1); i = i + iPrint)
			{
				cout << "[" << i << "," << j << "," << k << "]=" << scientific << relTime.t2[index(i, j, k, gs)] << "	";
			}
			cout << endl;
		}
	}
	cout << endl;
}

void NmrRelaxationSolve::ReadAndFillInAll()
{
	ReadStart();	//MPI++
	ReadActiveCells();	//MPI++
	BoundaryConditionsForActiveCells(); //MPI++
	ReadConcentration(); //MPI++
	ReadVelocity(); //MPI++ check
	ReadSurfRel();
	FillInDiffusinAndTime(); ////MPI++ check
	BoundaryConditionsForDiffusion(); //MPI++
	FillInMainField(); //ok
	FillInMagnetization(); //ok
	StepTime(); //ok
}

void NmrRelaxationSolve::DebugPrint()
{
	if (MPI_rank == 0)
	{
		PrintActiveCells();
		PrintConcentration();
		PrintDiffusionAndRelTime();
		cout << "Main Field" << endl;
		PrintVectorField(0, &mainField);
		cout << "Magnetization" << endl;
		PrintVectorField(0, &magnetization);
		cout << "Additional Magnetization" << endl;
		PrintVectorField(0, &addMagnetization);
	}
}

void NmrRelaxationSolve::FillInMagnetization()
{
	double pEq = equilibriumMagnetization;
	//for (int k = 1; k < (gs.kk + 2); k++)
	//{
	//	for (int j = 1; j < (gs.jk + 2); j++)
	//	{
	//		for (int i = 1; i < (gs.ik + 2); i++)
	//		{
	for (int k = 0 ; k < (gs.kk + 2); k++)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				if (actCell[index(i,j,k,gs)]=='1')
				{
					magnetization.x[index(i, j, k, gs)] = 0.0;
					magnetization.y[index(i, j, k, gs)] = 0.0;
					magnetization.z[index(i, j, k, gs)] = 0.0;
					//magnetization.x[index(i, j, k, gs)] = pEq;
					if (nDirectionField == 1) magnetization.x[index(i, j, k, gs)] = pEq;
					if (nDirectionField == 2) magnetization.y[index(i, j, k, gs)] = pEq;
					if (nDirectionField == 3) magnetization.z[index(i, j, k, gs)] = pEq;
					if (nDirectionField == 0)
					{
						double b_total = sqrt(bsx*bsx + bsy*bsy + bsz*bsz);
						double bnx = bsx / b_total, bny = bsy / b_total, bnz = bsz / b_total;
						magnetization.x[index(i, j, k, gs)] = pEq*bnx;
						magnetization.y[index(i, j, k, gs)] = pEq*bny;
						magnetization.z[index(i, j, k, gs)] = pEq*bnz;
					}
				}
				else
				{
					magnetization.x[index(i, j, k, gs)] = 0.0;
					magnetization.y[index(i, j, k, gs)] = 0.0;
					magnetization.z[index(i, j, k, gs)] = 0.0;
				}
				addMagnetization.x[index(i, j, k, gs)] = magnetization.x[index(i, j, k, gs)];
				addMagnetization.y[index(i, j, k, gs)] = magnetization.y[index(i, j, k, gs)];
				addMagnetization.z[index(i, j, k, gs)] = magnetization.z[index(i, j, k, gs)];
			}
		}
	}
}

void NmrRelaxationSolve::StepTime()
{
	if (sequenceTime1 != 0) dTime = min(min(min(dTimeNoise,dTimeGrad), min(dTimeVel, dTimeDiff)), min(dTimeRel, sequenceTime1 / 100.0));
	else dTime = min(min(min(dTimeNoise, dTimeGrad), min(dTimeVel, dTimeDiff)), min(dTimeRel, sequenceTime2 / 100.0));
	/*if ((dTimeVel <= dTimeDiff) && (dTimeVel <= dTimeRel)) dTime = dTimeVel;
	if ((dTimeDiff <=  dTimeVel) && (dTimeDiff <= dTimeRel)) dTime = dTimeDiff;
	if ((dTimeRel <= dTimeDiff) && (dTimeRel <= dTimeVel)) dTime = dTimeRel;*/
	cout << "dTime=" << scientific << dTime << " rank=" << MPI_rank << endl;
	//system("pause");
	//if (nSequence == 2) if (dTime >= sequenceTime1*0.5) dTime = sequenceTime1 / 100.0;
	//dTime = 1.0e-6;
	cout << "dTime=" << scientific << dTime << " rank=" << MPI_rank << endl;
	nStep = int(endTime / dTime)+1;
	//nStep = 5;
	cout << "nStep=" << nStep << " rank=" << MPI_rank << endl;	
}

void NmrRelaxationSolve::OrtTransform(int i,int j, int k, double nx, double ny, double nz, VectorField* field, double angle)
{
	double vec0x=0.0, vec0y=0.0, vec0z=0.0;
	double c, s, scal, length;
	length = sqrt(pow(nx*nx, 2.0) + pow(ny*ny, 2.0) + pow(nz*nz, 2.0));
	c = cos(angle);
	s = sin(angle);
	vec0x = field->x[index(i, j, k, gs)];
	vec0y = field->y[index(i, j, k, gs)];
	vec0z = field->z[index(i, j, k, gs)];
	scal = nx*vec0x+ ny*vec0y + nz*vec0z;
	field->x[index(i, j, k, gs)] = vec0x*c + nx*(1.0 - c)*scal + s*(ny*vec0z - nz*vec0y);
	field->y[index(i, j, k, gs)] = vec0y*c + ny*(1.0 - c)*scal + s*(nz*vec0x - nx*vec0z);
	field->z[index(i, j, k, gs)] = vec0z*c + nz*(1.0 - c)*scal + s*(nx*vec0y - ny*vec0x);
}

void NmrRelaxationSolve::AddImpulse(int flag, double grad, double nx, double ny, double nz, VectorField* field)
{
	double angle, length, distance;
	double nx0, ny0, nz0;
	double effectiveGradient;
	length = pow(nx*nx + ny*ny + nz*nz, 0.5);
	if (length == 0.0) length = 1.0;
	//cout << "nx = " << nx << " ny = " << ny << " nz = " << nz << " Length = " << length << endl;
	nx0 = double(nx) / length;
	ny0 = double(ny) / length;
	nz0 = double(nz) / length;
	distance = 10e+12;
	for (int k = 0; k < (gs.kk + 2); k++)
	{
		for (int j = 0; j < (gs.jk + 2); j++)
		{
			for (int i = 0; i < (gs.ik + 2); i++)
			{
				if (actCell[index(i, j, k, gs)] == '1')
				{
					//if (flag == 1) angle = grad*g*((i - 0.5 - int(gs.ik / 2))*dx*nx + (j - 0.5 - int(gs.jk / 2))*dy*ny + (k - 0.5 - int(gs.kk / 2))*dz*nz);
					if (flag == 1)
					{
						angle = grad*g*((i-1)*dx*nx + (j-1)*dy*ny + (k-1)*dz*nz);
						OrtTransform(i, j, k, nx0, ny0, nz0, field, angle);
					}
					if (flag == 2)
					{
						angle = grad;
						OrtTransform(i, j, k, nx, ny, nz, field, angle);
					}
					if (flag == 3)
					{
						
						effectiveGradient = pow(nx*nx + ny*ny + nz*nz, 0.5);
						angle = g*effectiveGradient*pow(pow((double(i) - 1.0 - distance)*dx*nx0 + (double(j) - 1.0 - distance)*dy*ny0 + (double(k) - 1.0 - distance)*dz*nz0,2),0.5);
						//sqrt(pow((i - 0.5*gs.ik + distance*nx0)*dx*nx0, 2.0) + pow((j - 0.5*gs.jk + distance*ny0)*dy*ny0, 2.0) + pow((k - 0.5*gs.kk + distance*nz0)*dz*nz0, 2.0));
						//angle = g*grad*sqrt(pow((i - 0.5*gs.ik + distance)*dx*nx, 2.0) + pow((j - 0.5*gs.jk + distance)*dy*ny, 2.0) + pow((k - 0.5*gs.kk + distance)*dz*nz, 2.0));
						//angle = g*grad*sqrt(pow((i - 1)*dx*nx, 2.0) + pow((j - 1)*dy*ny, 2.0) + pow((k - 1)*dz*nz, 2.0));
						OrtTransform(i, j, k, nx0, ny0, nz0, field, angle);
					}
					if (flag == 4)
					{
						angle = 0.0 - g*sqrt(pow((i - 0.5)*dx*nx - 0.5*gs.ik*dx*nx, 2.0) + pow((j - 0.5)*dy*ny - 0.5*gs.jk*dy*ny, 2.0) + pow((k - 0.5)*dz*nz - 0.5*gs.kk*dz*nz, 2.0));
						OrtTransform(i, j, k, nx, ny, nz, field, angle);
					}
					if ((i > 0 && i < gs.ik+1 )&& j==100/*(j>0 && j<gs.jk+1) */ /*&& (k>0 && k<gs.kk+1)*/)
					{
						//cout << "Gradient = " <<effectiveGradient<<" nx = "<< nx0 << " ny = "<< ny0 << " nz = " << nz0 <<" Angle = " << angle <<  endl;
					}
					
				}
			}
		}
	}
	//system("Pause");
}

void NmrRelaxationSolve::Sequence()
{
	//FID
	if (nSequence == 0)
	{
		if ((currentTime < dTime) && (currentTime >= 0.0))
		{
			AddImpulse(2, 0.5*pi, direction[2], direction[0], direction[1], &magnetization);
			cout << "90 pulse added" << endl;
		}

	}
	//CPMG 100 impulses
	if (nSequence == 1)
	{
		if ((currentTime < dTime) && (currentTime >= 0.0)) AddImpulse(2, 0.5*pi, direction[2], direction[0], direction[1], &magnetization);
		cout << "90 pulse added" << endl;
		for (int i = 0; i < 100; i++)
		{
			if ((currentTime < (2 * i + 1)*sequenceTime1 + dTime) && (currentTime >= (2 * i + 1)*sequenceTime1)) AddImpulse(2, pi, direction[2], direction[0], direction[1], &magnetization);
			cout << "180 pulse added" << endl;
		}
	}
	//PFG time1 - grad - time1 - 90 - time2 - 90 - time1 - grad - time1 - echo
	if (nSequence == 2)
	{
		if ((currentTime < dTime) && (currentTime >= 0.0) && (flag1 == 0)) 
		{
			AddImpulse(2, 0.5*pi, direction[2], direction[0], direction[1], &magnetization);
			//cout << "90 pulse added" << endl;
			flag1 = 1;
		}
		if ((currentTime < sequenceTime1 + dTime) && (currentTime >= sequenceTime1) && (flag2 == 0)) 
		{
			AddImpulse(1, gradientPulse, direction[0], direction[1], direction[2], &magnetization);
			//cout << "Gradient pulse added. " << "Time=" << currentTime << endl;
			flag2 = 1;
		}
		if ((currentTime < 2.0*sequenceTime1 + dTime) && (currentTime >= 2.0*sequenceTime1) && (flag3 == 0)) 
		{
			AddImpulse(2, 0.5*pi, direction[2], direction[0], direction[1], &magnetization);
			//cout << "90 pulse added" << endl;
			flag3 = 1;
		}
		if ((currentTime < 2.0*sequenceTime1 + sequenceTime2 + dTime) && (currentTime >= 2.0*sequenceTime1 + sequenceTime2) && (flag4 == 0))
		{
			AddImpulse(2, 0.5*pi, direction[2], direction[0], direction[1], &magnetization);
			//cout << "90 pulse added" << endl;
			flag4 = 1;
		}
		if ((currentTime < 3.0*sequenceTime1 + sequenceTime2 + dTime) && (currentTime >= 3.0*sequenceTime1 + sequenceTime2) && (flag5 == 0))
		{
			AddImpulse(1, gradientPulse, direction[0], direction[1], direction[2], &magnetization);
			//cout << "Gradient pulse added" << endl;
			flag5 = 1;
		}
		if ((currentTime < 3.5*sequenceTime1 + sequenceTime2 + dTime) && (currentTime >= 3.5*sequenceTime1 + sequenceTime2))
		{
			if ((flag1 == 0) || (flag2 == 0) || (flag3 == 0) || (flag4 == 0) || (flag5 == 0))
			{
				cout << "Sequence fail" << endl;
				system("pause");
			}
			else
			{
				//cout << "Sequence OK" << endl;
			}
		}
	}
}

void NmrRelaxationSolve::Sequence_gpu(double* magnx_d, double* magny_d, double* magnz_d)
{
	//FID
	if (nSequence == 0)
	{
		if ((currentTime < dTime) && (currentTime >= 0.0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(2, 0.5*pi, direction[2], direction[0], direction[1], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));

			cout << "90 pulse added" << endl;
		}

	}
	//CPMG 100 impulses
	if (nSequence == 1)
	{
		if ((currentTime < dTime) && (currentTime >= 0.0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(2, 0.5*pi, direction[2], direction[0], direction[1], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));


		}
		//cout << "90 pulse added" << endl;
		for (int i = 0; i < 20; i++)
		{
			if ((currentTime < (2 * i + 1)*sequenceTime1 + dTime) && (currentTime >= (2 * i + 1)*sequenceTime1))
			{
				gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
				gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
				gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

				AddImpulse(2, pi, direction[2], direction[0], direction[1], &magnetization);

				gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));

			}
			//cout << "180 pulse added" << endl;
		}
	}
	//PFG time1 - grad - time1 - 90 - time2 - 90 - time1 - grad - time1 - echo
	if (nSequence == 2)
	{
		if ((currentTime < dTime) && (currentTime >= 0.0) && (flag1 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(2, 0.5*pi, -direction[2], direction[0], direction[1], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "90 pulse added" << endl;
			flag1 = 1;
		}
		if ((currentTime < sequenceTime1 + dTime) && (currentTime >= sequenceTime1) && (flag2 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			/*char buf_f[250];
			sprintf(buf_f, "field_before_grad_%d_", PulseNum);
			string fileNamePattern = buf_f;
			WriteVectorFieldSeparated(&magnetization, fileNamePattern);*/

			AddImpulse(1, gradientPulse, direction[0], direction[1], direction[2], &magnetization);

			/*sprintf(buf_f, "field_after_grad_%d_", PulseNum);
			fileNamePattern = buf_f;
			WriteVectorFieldSeparated(&magnetization, fileNamePattern);*/

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));

			//cout << "Gradient pulse added. " << "Time=" << currentTime << endl;
			flag2 = 1;
		}
		if ((currentTime < 2.0*sequenceTime1 + dTime) && (currentTime >= 2.0*sequenceTime1) && (flag3 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			/*char buf_f[250];
			sprintf(buf_f, "field_before_90_%d_", PulseNum);
			string fileNamePattern = buf_f;
			WriteVectorFieldSeparated(&magnetization, fileNamePattern);*/

			AddImpulse(2, 0.5*pi, direction[2], direction[0], direction[1], &magnetization);

			/*sprintf(buf_f, "field_after_90_%d_", PulseNum);
			fileNamePattern = buf_f;
			WriteVectorFieldSeparated(&magnetization, fileNamePattern);*/

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));

			//cout << "90 pulse added" << endl;
			flag3 = 1;
		}
		if ((currentTime < 2.0*sequenceTime1 + sequenceTime2 + dTime) && (currentTime >= 2.0*sequenceTime1 + sequenceTime2) && (flag4 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(2, 0.5*pi, -direction[2], direction[0], direction[1], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));

			//cout << "90 pulse added" << endl;
			flag4 = 1;
		}
		if ((currentTime < 3.0*sequenceTime1 + sequenceTime2 + dTime) && (currentTime >= 3.0*sequenceTime1 + sequenceTime2) && (flag5 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(1, gradientPulse, direction[0], direction[1], direction[2], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "Gradient pulse added" << endl;
			flag5 = 1;
		}
		if ((currentTime < 3.5*sequenceTime1 + sequenceTime2 + dTime) && (currentTime >= 3.5*sequenceTime1 + sequenceTime2))
		{
			if ((flag1 == 0) || (flag2 == 0) || (flag3 == 0) || (flag4 == 0) || (flag5 == 0))
			{
				cout << "Sequence fail" << endl;
				cout << "Flag_1 = " << flag1 << " Flag_2 = " << flag2 << " Flag_3 = " << flag3 << " Flag_4 = " << flag4 << " Flag_5 = " << flag5 << endl;
				system("pause");
			}
			else
			{
				//cout << "Sequence OK" << endl;
			}
		}
	}
	//Basic gradients
	if (nSequence == 9)
	{
		if ((currentTime < dTime) && (currentTime >= 0.0) && (flag1 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(2, 0.5*pi, direction[2], direction[0], direction[1], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "90 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag1 = 1;
		}

		if ((currentTime < sequenceTime1 + dTime) && (currentTime >= sequenceTime1) && (flag2 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(1, gradientPulse, direction[0], direction[1], direction[2], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "90 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag2 = 1;
		}

		if ((currentTime < sequenceTime1 + sequenceTime2 / 2 + dTime) && (currentTime >= sequenceTime1 + sequenceTime2 / 2) && (flag3 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(2, pi, direction[2], direction[0], direction[1], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "90 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag3 = 1;
		}
		if ((currentTime < sequenceTime2 + sequenceTime1 + dTime) && (currentTime >= sequenceTime2 + sequenceTime1) && (flag4 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(1, gradientPulse, direction[0], direction[1], direction[2], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));

			//cout << "180 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag4 = 1;
		}
		if (currentTime >= sequenceTime2 + 2.05*sequenceTime1)
		{
			if ((flag1 == 0) || (flag2 == 0) || (flag3 == 0) || (flag4 == 0))
			{
				cout << "Sequence fail" << endl;
				cout << "Flag_1 = " << flag1 << " Flag_2 = " << flag2 << "Flag_3 = " << flag3 << " Flag_4 = " << flag4 << endl;
				system("pause");
			}
			else
			{
				//cout << "Sequence OK" << endl;
			}
		}
		
	}
	if (nSequence == 5)
	{
		if ((currentTime < dTime) && (currentTime >= 0.0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(2, pi, direction[2], direction[0], direction[1], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));

			cout << "180 pulse added" << endl;
		}
	}

	if (nSequence == 99)
	{
		double multiplier;
		multiplier = 10.0;
		if ((currentTime < dTime) && (currentTime >= 0.0) && (flag1 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			//AddImpulse(2, 0.5*pi, direction[2], direction[0], direction[1], &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "90 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag1 = 1;
		}

		if ((currentTime < sequenceTime1 + dTime) && (currentTime >= sequenceTime1) && (flag2 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(3, 1, multipleGradients[0], multipleGradients[1], multipleGradients[2], &magnetization);

			//for (int i = 0; i < int(multiplier); i++)
			//{
			//	AddImpulse(1, multipleGradients[0], 1, 0, 0, &magnetization);
			//	AddImpulse(1, multipleGradients[1] / multiplier, 0, 1, 0, &magnetization);
			//	AddImpulse(1, multipleGradients[2] / multiplier, 0, 0, 1, &magnetization);
			//}
			
			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "90 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag2 = 1; 
		}

		if ((currentTime < sequenceTime1 + sequenceTime2 / 2 + dTime) && (currentTime >= sequenceTime1 + sequenceTime2 / 2) && (flag3 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			//AddImpulse(2, pi*double(PulseNum[0]), 0, 1, 0, &magnetization);
			//AddImpulse(2, pi*double(PulseNum[1]), 0, 0, 1, &magnetization);
			//AddImpulse(2, pi*double(PulseNum[2]), 1, 0, 0, &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "90 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag3 = 1;
		}
		if ((currentTime < sequenceTime2 + sequenceTime1 + dTime) && (currentTime >= sequenceTime2 + sequenceTime1) && (flag4 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(3, 1, -multipleGradients[0], -multipleGradients[1], -multipleGradients[2], &magnetization);
			//AddImpulse(4, 1, multipleGradients[0], multipleGradients[1], multipleGradients[2], &magnetization);

			//for (int i = 0; i < int(multiplier); i++)
			//{
			//	AddImpulse(1, -multipleGradients[2] / multiplier, 0, 0, 1, &magnetization);
			//	AddImpulse(1, -multipleGradients[1] / multiplier, 0, 1, 0, &magnetization);
			//	AddImpulse(1, -multipleGradients[0], 1, 0, 0, &magnetization);
			//}

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));

			//cout << "180 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag4 = 1;
		}
		if (currentTime >= sequenceTime2 + 2.05*sequenceTime1)
		{
			if ((flag1 == 0) || (flag2 == 0) || (flag3 == 0) || (flag4 == 0))
			{
				cout << "Sequence fail" << endl;
				cout << "Flag_1 = " << flag1 << " Flag_2 = " << flag2 << "Flag_3 = " << flag3 << " Flag_4 = " << flag4 << endl;
				system("pause");
			}
			else
			{
				//cout << "Sequence OK" << endl;
			}
		}

	}

	if (nSequence == 98)
	{
		double multiplier;
		multiplier = 10.0;
		if ((currentTime < dTime) && (currentTime >= 0.0) && (flag1 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(2, 0.5*pi, 0, -1, 0, &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "90 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag1 = 1;
		}

		if ((currentTime < sequenceTime1 + dTime) && (currentTime >= sequenceTime1) && (flag2 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(3, 1, multipleGradients[0], multipleGradients[1], multipleGradients[2], &magnetization);

			//for (int i = 0; i < int(multiplier); i++)
			//{
			//	AddImpulse(1, multipleGradients[0], 1, 0, 0, &magnetization);
			//	AddImpulse(1, multipleGradients[1] / multiplier, 0, 1, 0, &magnetization);
			//	AddImpulse(1, multipleGradients[2] / multiplier, 0, 0, 1, &magnetization);
			//}

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "90 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag2 = 1;
		}

		if ((currentTime < sequenceTime1 + sequenceTime2 / 2 + dTime) && (currentTime >= sequenceTime1 + sequenceTime2 / 2) && (flag3 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(2, pi, 0, 0, 1, &magnetization);
			//AddImpulse(2, pi*double(PulseNum[0]), 0, 1, 0, &magnetization);
			//AddImpulse(2, pi*double(PulseNum[1]), 0, 0, 1, &magnetization);
			//AddImpulse(2, pi*double(PulseNum[2]), 1, 0, 0, &magnetization);

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			//cout << "90 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag3 = 1;
		}
		if ((currentTime < sequenceTime2 + sequenceTime1 + dTime) && (currentTime >= sequenceTime2 + sequenceTime1) && (flag4 == 0))
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			AddImpulse(3, 1, multipleGradients[0], multipleGradients[1], multipleGradients[2], &magnetization);
			//AddImpulse(4, 1, multipleGradients[0], multipleGradients[1], multipleGradients[2], &magnetization);

			//for (int i = 0; i < int(multiplier); i++)
			//{
			//	AddImpulse(1, -multipleGradients[2] / multiplier, 0, 0, 1, &magnetization);
			//	AddImpulse(1, -multipleGradients[1] / multiplier, 0, 1, 0, &magnetization);
			//	AddImpulse(1, -multipleGradients[0], 1, 0, 0, &magnetization);
			//}

			gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magny_d, magnetization.y, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyHostToDevice));

			//cout << "180 pulse added" << endl;
			//cout << "Current time = " << currentTime << endl;
			flag4 = 1;
		}
		if (currentTime >= sequenceTime2 + 2.05*sequenceTime1)
		{
			if ((flag1 == 0) || (flag2 == 0) || (flag3 == 0) || (flag4 == 0))
			{
				cout << "Sequence fail" << endl;
				cout << "Flag_1 = " << flag1 << " Flag_2 = " << flag2 << "Flag_3 = " << flag3 << " Flag_4 = " << flag4 << endl;
				system("pause");
			}
			else
			{
				//cout << "Sequence OK" << endl;
			}
		}

	}

}

void NmrRelaxationSolve::ExplicitSimple()
{
	double hx, hy, hz, hxx, hyy, hzz;
	int km, kp, jm, jp, im, ip;
	double tau1, tau2, dxm, dxp, dym, dyp, dzm, dzp, u, v, w, ud, vd, wd, up, upp, um, umm, vp, vpp, vm, vmm, wp, wpp, wm, wmm;
	double px0, py0, pz0, anb, anx, any, anz, div, divx, divy, divz, pEq, pxEq, pyEq, pzEq;
	double b, bbxb, bbyb, bbzb;
	hx = dTime / dx;
	hy = dTime / dy;
	hz = dTime / dz;
	hxx = dTime / (dx*dx);
	hyy = dTime / (dy*dy);
	hzz = dTime / (dz*dz);

	for (int k = 1; k < (gs.kk+1); k++)
	{
		km = k - 1;
		kp = k + 1;
		for (int j = 1; j < (gs.jk + 1); j++)
		{
			jm = j - 1;
			jp = j + 1;
			for (int i = 1; i < (gs.ik+1); i++)
			{
				im = i - 1;
				ip = i + 1;
				if (actCell[index(i,j,k,gs)] == '1')
				{
					tau1 = 1.0 / relTime.t1[index(i, j, k, gs)];
					tau2 = 1.0 / relTime.t2[index(i, j, k, gs)];
					dxp = 2.0*(diffusion[index(ip, j, k, gs)] * diffusion[index(i, j, k, gs)]) / (diffusion[index(ip, j, k, gs)] + diffusion[index(i, j, k, gs)] + small);
					dxm = 2.0*(diffusion[index(im, j, k, gs)] * diffusion[index(i, j, k, gs)]) / (diffusion[index(im, j, k, gs)] + diffusion[index(i, j, k, gs)] + small);
					dyp = 2.0*(diffusion[index(i, jp, k, gs)] * diffusion[index(i, j, k, gs)]) / (diffusion[index(i, jp, k, gs)] + diffusion[index(i, j, k, gs)] + small);
					dym = 2.0*(diffusion[index(i, jm, k, gs)] * diffusion[index(i, j, k, gs)]) / (diffusion[index(i, jm, k, gs)] + diffusion[index(i, j, k, gs)] + small);
					dzp = 2.0*(diffusion[index(i, j, kp, gs)] * diffusion[index(i, j, k, gs)]) / (diffusion[index(i, j, kp, gs)] + diffusion[index(i, j, k, gs)] + small);
					dzm = 2.0*(diffusion[index(i, j, km, gs)] * diffusion[index(i, j, k, gs)]) / (diffusion[index(i, j, km, gs)] + diffusion[index(i, j, k, gs)] + small);
					px0 = magnetization.x[index(i, j, k, gs)];
					py0 = magnetization.y[index(i, j, k, gs)];
					pz0 = magnetization.z[index(i, j, k, gs)];
					um = velocity.x[index(i, j, k, gs)];
					umm = um / (um + small);
					vm = velocity.y[index(i, j, k, gs)];
					vmm = vm / (vm + small);
					wm = velocity.z[index(i, j, k, gs)];
					wmm = wm / (wm + small);
					up = velocity.x[index(ip, j, k, gs)];
					upp = up / (up + small);
					vp = velocity.y[index(i, jp, k, gs)];
					vpp = vp / (vp + small);
					wp = velocity.z[index(i, j, kp, gs)];
					wpp = wp / (wp + small);
					u = 0.5*(um + up);
					v = 0.5*(vm + vp);
					w = 0.5*(wm + wp);
					ud = abs(u);
					vd = abs(v);
					wd = abs(w);
					anb = sqrt(mainField.x[index(i, j, k, gs)] * mainField.x[index(i, j, k, gs)] + mainField.y[index(i, j, k, gs)] * mainField.y[index(i, j, k, gs)] + mainField.z[index(i, j, k, gs)] * mainField.z[index(i, j, k, gs)]);
					anx = mainField.x[index(i, j, k, gs)] / anb;
					any = mainField.y[index(i, j, k, gs)] / anb;
					anz = mainField.z[index(i, j, k, gs)] / anb;
					pEq = equilibriumMagnetization;
					pxEq = pEq*anx;
					pyEq = pEq*any;
					pzEq = pEq*anz;
					div = anx*px0 + any*py0 + anz*pz0;
					divx = anx*div;
					divy = any*div;
					divz = anz*div;

					b = dTime*g;
					bbxb = b*noiseField.x[index(i, j, k, gs)];
					bbyb = b*noiseField.y[index(i, j, k, gs)];
					bbzb = b*noiseField.z[index(i, j, k, gs)];
					//Solver itself
					addMagnetization.x[index(i, j, k, gs)] = px0 + dTime*tau1*(pxEq - divx) + dTime*tau2*(divx - px0);
					addMagnetization.y[index(i, j, k, gs)] = py0 + dTime*tau1*(pyEq - divy) + dTime*tau2*(divy - py0);
					addMagnetization.z[index(i, j, k, gs)] = pz0 + dTime*tau1*(pzEq - divz) + dTime*tau2*(divz - pz0);
					//cout << "dTime=" << dTime << endl << "tau1=" << tau1 << endl << "(pxEq - divx)=" << (pxEq - divx) << endl;
					//system("pause");
					
					//Add smth with flags
					if (nVectorMult == 1)
					{
						addMagnetization.x[index(i, j, k, gs)] += 0.0 - bbyb*pz0 + bbzb*py0;
						addMagnetization.y[index(i, j, k, gs)] += 0.0 + bbxb*pz0 - bbzb*px0;
						addMagnetization.z[index(i, j, k, gs)] += 0.0 - bbxb*py0 + bbyb*px0;
					}
					if (nVelocity == 1)
					{
						addMagnetization.x[index(i, j, k, gs)] -= 0.5*hx*((u - ud)*(magnetization.x[index(ip, j, k, gs)] - px0)*upp + (u + ud)*(px0 - magnetization.x[index(im, j, k, gs)])*umm)
							+ 0.5*hy*((v - vd)*(magnetization.x[index(i, jp, k, gs)] - px0)*vpp + (v + vd)*(px0 - magnetization.x[index(i, jm, k, gs)])*vmm)
							+ 0.5*hz*((w - wd)*(magnetization.x[index(i, j, kp, gs)] - px0)*wpp + (w + wd)*(px0 - magnetization.x[index(i, j, km, gs)])*wmm);
						addMagnetization.y[index(i, j, k, gs)] -= 0.5*hx*((u - ud)*(magnetization.y[index(ip, j, k, gs)] - py0)*upp + (u + ud)*(py0 - magnetization.y[index(im, j, k, gs)])*umm)
							+ 0.5*hy*((v - vd)*(magnetization.y[index(i, jp, k, gs)] - py0)*vpp + (v + vd)*(py0 - magnetization.y[index(i, jm, k, gs)])*vmm)
							+ 0.5*hz*((w - wd)*(magnetization.y[index(i, j, kp, gs)] - py0)*wpp + (w + wd)*(py0 - magnetization.y[index(i, j, km, gs)])*wmm);
						addMagnetization.z[index(i, j, k, gs)] -= 0.5*hx*((u - ud)*(magnetization.z[index(ip, j, k, gs)] - pz0)*upp + (u + ud)*(pz0 - magnetization.z[index(im, j, k, gs)])*umm)
							+ 0.5*hy*((v - vd)*(magnetization.z[index(i, jp, k, gs)] - pz0)*vpp + (v + vd)*(pz0 - magnetization.z[index(i, jm, k, gs)])*vmm)
							+ 0.5*hz*((w - wd)*(magnetization.z[index(i, j, kp, gs)] - pz0)*wpp + (w + wd)*(pz0 - magnetization.z[index(i, j, km, gs)])*wmm);
					}
					if (nDiffusion  == 1)
					{
						addMagnetization.x[index(i, j, k, gs)] += hxx*(dxp*(magnetization.x[index(ip, j, k, gs)] - px0) - dxm*(px0 - magnetization.x[index(im, j, k, gs)]))
							+ hyy*(dyp*(magnetization.x[index(i, jp, k, gs)] - px0) - dym*(px0 - magnetization.x[index(i, jm, k, gs)]))
							+ hzz*(dzp*(magnetization.x[index(i, j, kp, gs)] - px0) - dzm*(px0 - magnetization.x[index(i, j, km, gs)]));
						addMagnetization.y[index(i, j, k, gs)] += hxx*(dxp*(magnetization.y[index(ip, j, k, gs)] - py0) - dxm*(py0 - magnetization.y[index(im, j, k, gs)]))
							+ hyy*(dyp*(magnetization.y[index(i, jp, k, gs)] - py0) - dym*(py0 - magnetization.y[index(i, jm, k, gs)]))
							+ hzz*(dzp*(magnetization.y[index(i, j, kp, gs)] - py0) - dzm*(py0 - magnetization.y[index(i, j, km, gs)]));
						addMagnetization.z[index(i, j, k, gs)] += hxx*(dxp*(magnetization.z[index(ip, j, k, gs)] - pz0) - dxm*(pz0 - magnetization.z[index(im, j, k, gs)]))
							+ hyy*(dyp*(magnetization.z[index(i, jp, k, gs)] - pz0) - dym*(pz0 - magnetization.z[index(i, jm, k, gs)]))
							+ hzz*(dzp*(magnetization.z[index(i, j, kp, gs)] - pz0) - dzm*(pz0 - magnetization.z[index(i, j, km, gs)]));
					}
				}
			}
		}
	}

	//memcpy(magnetization, addMagnetization, (gs.ik+2)*(gs.jk+2)*(gs.kk+2)*sizeof(VectorField));
	//memcpy(magnetization.x, addMagnetization.x, (gs.ik+2)*(gs.jk+2)*(gs.kk+2)*sizeof(double));
	//memcpy(magnetization.y, addMagnetization.y, gs.ik*gs.jk*gs.kk*sizeof(double));
	//memcpy(magnetization.z, addMagnetization.z, gs.ik*gs.jk*gs.kk*sizeof(double));
	for (int k = 1; k < (gs.kk + 1); k++)
	{
		for (int j = 1; j < (gs.jk + 1); j++)
		{
			for (int i = 1; i < (gs.ik + 1); i++)
			{
				magnetization.x[index(i, j, k, gs)] = addMagnetization.x[index(i, j, k, gs)];
				magnetization.y[index(i, j, k, gs)] = addMagnetization.y[index(i, j, k, gs)];
				magnetization.z[index(i, j, k, gs)] = addMagnetization.z[index(i, j, k, gs)];
			}
		}
	}

}

int NmrRelaxationSolve::index(int i, int j, int k, GridSize gs)
{
	return i + j*(gs.ik + 2) + k*(gs.ik + 2)*(gs.jk + 2);
}

double NmrRelaxationSolve::min(double a, double b)
{
	if (a <= b) return a;
	else return b;
}

double NmrRelaxationSolve::max(double a, double b)
{
	if (a <= b) return b;
	else return a;
}

void NmrRelaxationSolve::OutputX(VectorField* field)
{
	double sum = 0.0,z1,z2,c1,c2, sum_1=0.0, sum_2=0.0;
	int test = 0;
	if (nComponents == 1)
	{
		for (int k = 1 + bufferSizeZ; k < (gs.kk + 1 - bufferSizeZ); k++)
		{
			for (int j = 1 + bufferSizeY; j < (gs.jk + 1 - bufferSizeY); j++)
			{
				for (int i = 1 + bufferSizeX; i < (gs.ik + 1 - bufferSizeX); i++)
				{
					if (actCell[index(i, j, k, gs)] == '1')
					{
						sum += field->x[index(i, j, k, gs)];
						test += 1;
					}
				}
			}
		}
		MPI_Reduce(&sum, &resX, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}

	if (nComponents == 2)
	{
		for (int k = 1 + bufferSizeZ; k < (gs.kk + 1 - bufferSizeZ); k++)
		{
			for (int j = 1 + bufferSizeY; j < (gs.jk + 1 - bufferSizeY); j++)
			{
				for (int i = 1 + bufferSizeX; i < (gs.ik + 1 - bufferSizeX); i++)
				{
					if (actCell[index(i, j, k, gs)] == '1')
					{
						z1 = sqrt((concentration[0][index(i, j, k, gs)] - an1) * (concentration[0][index(i, j, k, gs)] - an1) + (concentration[1][index(i, j, k, gs)] - an2) * (concentration[1][index(i, j, k, gs)] - an2));
						z2 = sqrt((concentration[0][index(i, j, k, gs)] - bn1) * (concentration[0][index(i, j, k, gs)] - bn1) + (concentration[1][index(i, j, k, gs)] - bn2) * (concentration[1][index(i, j, k, gs)] - bn2));
						c1 = z2 / (z1 + z2);
						c2 = z1 / (z1 + z2);
						if (c1>0.8 || c2>0.8)
						{
							sum_1 += field->x[index(i, j, k, gs)] * c1;
							sum_2 += field->x[index(i, j, k, gs)] * c2;
							test += 1;
						}
					}
				}
			}
		}
		MPI_Reduce(&sum_1, &resX1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&sum_2, &resX2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	
	MPI_Reduce(&test, &nnx, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
}

void NmrRelaxationSolve::OutputY(VectorField* field)
{
	double sum = 0.0, z1, z2, c1, c2, sum_1 = 0.0, sum_2 = 0.0;
	int test = 0;
	if (nComponents == 1)
	{
		for (int k = 1 + bufferSizeZ; k < (gs.kk + 1 - bufferSizeZ); k++)
		{
			for (int j = 1 + bufferSizeY; j < (gs.jk + 1 - bufferSizeY); j++)
			{
				for (int i = 1 + bufferSizeX; i < (gs.ik + 1 - bufferSizeX); i++)
				{
					if (actCell[index(i, j, k, gs)] == '1')
					{
						sum += field->y[index(i, j, k, gs)];
						test += 1;
					}
				}
			}
		}
		MPI_Reduce(&sum, &resY, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}

	if (nComponents == 2)
	{
		for (int k = 1 + bufferSizeZ; k < (gs.kk + 1 - bufferSizeZ); k++)
		{
			for (int j = 1 + bufferSizeY; j < (gs.jk + 1 - bufferSizeY); j++)
			{
				for (int i = 1 + bufferSizeX; i < (gs.ik + 1 - bufferSizeX); i++)
				{
					if (actCell[index(i, j, k, gs)] == '1')
					{
						z1 = sqrt((concentration[0][index(i, j, k, gs)] - an1) * (concentration[0][index(i, j, k, gs)] - an1) + (concentration[1][index(i, j, k, gs)] - an2) * (concentration[1][index(i, j, k, gs)] - an2));
						z2 = sqrt((concentration[0][index(i, j, k, gs)] - bn1) * (concentration[0][index(i, j, k, gs)] - bn1) + (concentration[1][index(i, j, k, gs)] - bn2) * (concentration[1][index(i, j, k, gs)] - bn2));
						c1 = z2 / (z1 + z2);
						c2 = z1 / (z1 + z2);
						if (c1>0.8 || c2>0.8)
						{
							sum_1 += field->y[index(i, j, k, gs)] * c1;
							sum_2 += field->y[index(i, j, k, gs)] * c2;
							test += 1;
						}
					}
				}
			}
		}
		MPI_Reduce(&sum_1, &resY1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&sum_2, &resY2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}

	MPI_Reduce(&test, &nny, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
}

void NmrRelaxationSolve::OutputZ(VectorField* field)
{
	double sum = 0.0, z1, z2, c1, c2, sum_1 = 0.0, sum_2 = 0.0;
	int test = 0;
	if (nComponents == 1)
	{
		for (int k = 1 + bufferSizeZ; k < (gs.kk + 1 - bufferSizeZ); k++)
		{
			for (int j = 1 + bufferSizeY; j < (gs.jk + 1 - bufferSizeY); j++)
			{
				for (int i = 1 + bufferSizeX; i < (gs.ik + 1 - bufferSizeX); i++)
				{
					if (actCell[index(i, j, k, gs)] == '1')
					{
						sum += field->z[index(i, j, k, gs)];
						test += 1;
					}
				}
			}
		}
		MPI_Reduce(&sum, &resZ, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}

	if (nComponents == 2)
	{
		for (int k = 1 + bufferSizeZ; k < (gs.kk + 1 - bufferSizeZ); k++)
		{
			for (int j = 1 + bufferSizeY; j < (gs.jk + 1 - bufferSizeY); j++)
			{
				for (int i = 1 + bufferSizeX; i < (gs.ik + 1 - bufferSizeX); i++)
				{
					if (actCell[index(i, j, k, gs)] == '1')
					{
						z1 = sqrt((concentration[0][index(i, j, k, gs)] - an1) * (concentration[0][index(i, j, k, gs)] - an1) + (concentration[1][index(i, j, k, gs)] - an2) * (concentration[1][index(i, j, k, gs)] - an2));
						z2 = sqrt((concentration[0][index(i, j, k, gs)] - bn1) * (concentration[0][index(i, j, k, gs)] - bn1) + (concentration[1][index(i, j, k, gs)] - bn2) * (concentration[1][index(i, j, k, gs)] - bn2));
						c1 = z2 / (z1 + z2);
						c2 = z1 / (z1 + z2);
						if (c1>0.8 || c2>0.8)
						{
							sum_1 += field->z[index(i, j, k, gs)] * c1;
							sum_2 += field->z[index(i, j, k, gs)] * c2;
							test += 1;
						}
					}
				}
			}
		}
		MPI_Reduce(&sum_1, &resZ1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&sum_2, &resZ2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}

	MPI_Reduce(&test, &nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
}
