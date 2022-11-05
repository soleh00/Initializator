#include <cuda_runtime.h>
#include "../headers/NmrRelaxationSolve.h"

#define CUDA_THREAD_PER_BLOCK 128
#define BOR_BLOCK_SIZE 32

__device__  int indexDevice(int i, int j, int k, GridSize gs)
{
	return i+ (gs.ik +2)*j +(gs.ik +2)* (gs.jk +2) *k;
}

void NmrRelaxationSolve::InitializeGpu()
{
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	const int num_el = (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2);
	gpuErrchk(cudaMalloc((void **)&actCell_d, num_el*sizeof(char)));

	gpuErrchk(cudaMalloc((void **)&diff_d, num_el*sizeof(double)));

	gpuErrchk(cudaMalloc((void **)&magnx_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&magny_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&magnz_d, num_el*sizeof(double)));

	gpuErrchk(cudaMalloc((void **)&mFx_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&mFy_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&mFz_d, num_el*sizeof(double)));

	gpuErrchk(cudaMalloc((void **)&nFx_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&nFy_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&nFz_d, num_el*sizeof(double)));

	gpuErrchk(cudaMalloc((void **)&velx_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&vely_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&velz_d, num_el*sizeof(double)));

	gpuErrchk(cudaMalloc((void **)&addMagx_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&addMagy_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&addMagz_d, num_el*sizeof(double)));

	gpuErrchk(cudaMalloc((void **)&predictorX, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&predictorY, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&predictorZ, num_el*sizeof(double)));

	gpuErrchk(cudaMalloc((void **)&correctorX, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&correctorY, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&correctorZ, num_el*sizeof(double)));

	gpuErrchk(cudaMalloc((void **)&softnerX, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&softnerY, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&softnerZ, num_el*sizeof(double)));

	gpuErrchk(cudaMalloc((void **)&relT1_d, num_el*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&relT2_d, num_el*sizeof(double)));
	//
	cudaMemset(predictorX, 0.0, num_el*sizeof(double));
	cudaMemset(predictorY, 0.0, num_el*sizeof(double));
	cudaMemset(predictorZ, 0.0, num_el*sizeof(double));
	cudaMemset(correctorX, 0.0, num_el*sizeof(double));
	cudaMemset(correctorY, 0.0, num_el*sizeof(double));
	cudaMemset(correctorZ, 0.0, num_el*sizeof(double));

	/*CUDA_arrays set*/
	gpuErrchk(cudaMemcpy(actCell_d, actCell, num_el*sizeof(char), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(diff_d, diffusion, num_el*sizeof(double), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(magny_d, magnetization.y, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, num_el*sizeof(double), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(mFx_d, mainField.x, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(mFy_d, mainField.y, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(mFz_d, mainField.z, num_el*sizeof(double), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(nFx_d, noiseField.x, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(nFy_d, noiseField.y, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(nFz_d, noiseField.z, num_el*sizeof(double), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(velx_d, velocity.x, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(vely_d, velocity.y, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(velz_d, velocity.z, num_el*sizeof(double), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(addMagx_d, addMagnetization.x, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(addMagy_d, addMagnetization.y, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(addMagz_d, addMagnetization.z, num_el*sizeof(double), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(relT1_d, relTime.t1, num_el*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(relT2_d, relTime.t2, num_el*sizeof(double), cudaMemcpyHostToDevice));
}

__global__ void NMR_2d (char* actCell_d, double* diff_d, 
	double *magnx_d, double* magny_d, double* magnz_d,
	double* mFx_d, double* mFy_d, double* mFz_d,
	double* nFx_d, double* nFy_d, double* nFz_d,
	double* velx_d, double* vely_d, double* velz_d,
	double* addMagx_d, double* addMagy_d, double* addMagz_d,
	double dTime, double dx, double dy, double dz, GridSize gs,
	int nVectorMult, int nVelocity, int nDiffusion, double* relT1_d, double* relT2_d, double equilibriumMagnetization)
{
	double hx, hy, hz, hxx, hyy, hzz;
	int km, kp, jm, jp, im, ip;
	double fxp, fxm, fyp, fym, fzp, fzm;
	double tau1, tau2, dxm, dxp, dym, dyp, dzm, dzp, u, v, w, ud, vd, wd, up, upp, um, umm, vp, vpp, vm, vmm, wp, wpp, wm, wmm;
	double px0, py0, pz0, anb, anx, any, anz, div, divx, divy, divz, pEq, pxEq, pyEq, pzEq;
	double b, bbxb, bbyb, bbzb;
	hx = dTime / dx;
	hy = dTime / dy;
	hz = dTime / dz;
	hxx = dTime / (dx*dx);
	hyy = dTime / (dy*dy);
	hzz = dTime / (dz*dz);

	int thIdx=blockIdx.x * blockDim.x + threadIdx.x;

	//const int i = blockIdx.x * blockDim.x + threadIdx.x+ 1;
	//const int j = blockIdx.y * blockDim.y + threadIdx.y+ 1;
	//const int k = blockIdx.z * blockDim.z + threadIdx.z+ 1;
	//if ((i<gs.ik-1)&&(j<gs.jk-1)&&(k<gs.kk-1))

	const int gsni=gs.ik;
	const int gsnj=gs.jk;
	if (thIdx<gsni*gsnj)	
	{
		const int i = (thIdx%gsni)+1;
		const int j = (int)(thIdx/gsni)+1;
		const int k = blockIdx.y+1;

		km = k - 1;
		kp = k + 1;
		jm = j - 1;
		jp = j + 1;
		im = i - 1;
		ip = i + 1;
		if (actCell_d[indexDevice(i, j, k, gs)] == '1')
		{
			tau1 = 1.0 / relT1_d[indexDevice(i, j, k, gs)];//+
			tau2 = 1.0 / relT2_d[indexDevice(i, j, k, gs)];//+
			
			px0 = magnx_d[indexDevice(i, j, k, gs)];
			py0 = magny_d[indexDevice(i, j, k, gs)];
			pz0 = magnz_d[indexDevice(i, j, k, gs)];
			um = velx_d[indexDevice(i, j, k, gs)];
			umm = um / (um + 1e-20);
			vm = vely_d[indexDevice(i, j, k, gs)];
			vmm = vm / (vm + 1e-20);
			wm = velz_d[indexDevice(i, j, k, gs)];
			wmm = wm / (wm + 1e-20);
			up = velx_d[indexDevice(ip, j, k, gs)];
			upp = up / (up + 1e-20);
			vp = vely_d[indexDevice(i, jp, k, gs)];
			vpp = vp / (vp + 1e-20);
			wp = velz_d[indexDevice(i, j, kp, gs)];
			wpp = wp / (wp + 1e-20);
			u = 0.5*(um + up);
			v = 0.5*(vm + vp);
			w = 0.5*(wm + wp);
			ud = fabs(u);
			vd = fabs(v);
			wd = fabs(w);
			anb = sqrt(mFx_d[indexDevice(i, j, k, gs)] * mFx_d[indexDevice(i, j, k, gs)] + mFy_d[indexDevice(i, j, k, gs)] * mFy_d[indexDevice(i, j, k, gs)] + mFz_d[indexDevice(i, j, k, gs)] * mFz_d[indexDevice(i, j, k, gs)]);
			anx = mFx_d[indexDevice(i, j, k, gs)] / anb;
			any = mFy_d[indexDevice(i, j, k, gs)] / anb;
			anz = mFz_d[indexDevice(i, j, k, gs)] / anb;
			pEq = equilibriumMagnetization;
			pxEq = pEq*anx;
			pyEq = pEq*any;
			pzEq = pEq*anz;
			div = anx*px0 + any*py0 + anz*pz0;
			divx = anx*div;
			divy = any*div;
			divz = anz*div;

			b = dTime*4.78941714e+7;
			bbxb = b*nFx_d[indexDevice(i, j, k, gs)];
			bbyb = b*nFy_d[indexDevice(i, j, k, gs)];
			bbzb = b*nFz_d[indexDevice(i, j, k, gs)];
			//Solver itself
			addMagx_d[indexDevice(i, j, k, gs)] = px0 + dTime*tau1*(pxEq - divx) + dTime*tau2*(divx - px0);
			addMagy_d[indexDevice(i, j, k, gs)] = py0 + dTime*tau1*(pyEq - divy) + dTime*tau2*(divy - py0);
			addMagz_d[indexDevice(i, j, k, gs)] = pz0 + dTime*tau1*(pzEq - divz) + dTime*tau2*(divz - pz0);

			//Add smth with flags
			if (nVectorMult == 1)
			{
				addMagx_d[indexDevice(i, j, k, gs)] += 0.0 - bbyb*pz0 + bbzb*py0;
				addMagy_d[indexDevice(i, j, k, gs)] += 0.0 + bbxb*pz0 - bbzb*px0;
				addMagz_d[indexDevice(i, j, k, gs)] += 0.0 - bbxb*py0 + bbyb*px0;
			}
			if (nVelocity == 1)
			{
				if (actCell_d[indexDevice(im, j, k, gs)] == '0') fxm = 0.0;
				else fxm = 1.0;
				if (actCell_d[indexDevice(ip, j, k, gs)] == '0') fxp = 0.0;
				else fxp = 1.0;
				if (actCell_d[indexDevice(i, jm, k, gs)] == '0') fym = 0.0;
				else fym = 1.0;
				if (actCell_d[indexDevice(i, jp, k, gs)] == '0') fyp = 0.0;
				else fyp = 1.0;
				if (actCell_d[indexDevice(i, j, km, gs)] == '0') fzm = 0.0;
				else fzm = 1.0;
				if (actCell_d[indexDevice(i, j, kp, gs)] == '0') fzp = 0.0;
				else fzp = 1.0;
				addMagx_d[indexDevice(i, j, k, gs)] -= 0.5*hx*((u - ud)*(magnx_d[indexDevice(ip, j, k, gs)] - px0)*upp*fxp + (u + ud)*(px0 - magnx_d[indexDevice(im, j, k, gs)])*umm*fxm)
					+ 0.5*hy*((v - vd)*(magnx_d[indexDevice(i, jp, k, gs)] - px0)*vpp*fyp + (v + vd)*(px0 - magnx_d[indexDevice(i, jm, k, gs)])*vmm*fym)
					+ 0.5*hz*((w - wd)*(magnx_d[indexDevice(i, j, kp, gs)] - px0)*wpp*fzp + (w + wd)*(px0 - magnx_d[indexDevice(i, j, km, gs)])*wmm*fzm);
				addMagy_d[indexDevice(i, j, k, gs)] -= 0.5*hx*((u - ud)*(magny_d[indexDevice(ip, j, k, gs)] - py0)*upp*fxp + (u + ud)*(py0 - magny_d[indexDevice(im, j, k, gs)])*umm*fxm)
					+ 0.5*hy*((v - vd)*(magny_d[indexDevice(i, jp, k, gs)] - py0)*vpp*fyp + (v + vd)*(py0 - magny_d[indexDevice(i, jm, k, gs)])*vmm*fym)
					+ 0.5*hz*((w - wd)*(magny_d[indexDevice(i, j, kp, gs)] - py0)*wpp*fzp + (w + wd)*(py0 - magny_d[indexDevice(i, j, km, gs)])*wmm*fzm);
				addMagz_d[indexDevice(i, j, k, gs)] -= 0.5*hx*((u - ud)*(magnz_d[indexDevice(ip, j, k, gs)] - pz0)*upp*fxp + (u + ud)*(pz0 - magnz_d[indexDevice(im, j, k, gs)])*umm*fxm)
					+ 0.5*hy*((v - vd)*(magnz_d[indexDevice(i, jp, k, gs)] - pz0)*vpp*fyp + (v + vd)*(pz0 - magnz_d[indexDevice(i, jm, k, gs)])*vmm*fym)
					+ 0.5*hz*((w - wd)*(magnz_d[indexDevice(i, j, kp, gs)] - pz0)*wpp*fzp + (w + wd)*(pz0 - magnz_d[indexDevice(i, j, km, gs)])*wmm*fzm);
			}
			if (nDiffusion == 1)
			{
				dxp = 2.0*(diff_d[indexDevice(ip, j, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(ip, j, k, gs)] + diff_d[indexDevice(i, j, k, gs)]);
				dxm = 2.0*(diff_d[indexDevice(im, j, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(im, j, k, gs)] + diff_d[indexDevice(i, j, k, gs)]);
				dyp = 2.0*(diff_d[indexDevice(i, jp, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, jp, k, gs)] + diff_d[indexDevice(i, j, k, gs)]);
				dym = 2.0*(diff_d[indexDevice(i, jm, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, jm, k, gs)] + diff_d[indexDevice(i, j, k, gs)]);
				dzp = 2.0*(diff_d[indexDevice(i, j, kp, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, j, kp, gs)] + diff_d[indexDevice(i, j, k, gs)]);
				dzm = 2.0*(diff_d[indexDevice(i, j, km, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, j, km, gs)] + diff_d[indexDevice(i, j, k, gs)]);
				addMagx_d[indexDevice(i, j, k, gs)] += hxx*(dxp*(magnx_d[indexDevice(ip, j, k, gs)] - px0) - dxm*(px0 - magnx_d[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(magnx_d[indexDevice(i, jp, k, gs)] - px0) - dym*(px0 - magnx_d[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(magnx_d[indexDevice(i, j, kp, gs)] - px0) - dzm*(px0 - magnx_d[indexDevice(i, j, km, gs)]));
				addMagy_d[indexDevice(i, j, k, gs)] += hxx*(dxp*(magny_d[indexDevice(ip, j, k, gs)] - py0) - dxm*(py0 - magny_d[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(magny_d[indexDevice(i, jp, k, gs)] - py0) - dym*(py0 - magny_d[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(magny_d[indexDevice(i, j, kp, gs)] - py0) - dzm*(py0 - magny_d[indexDevice(i, j, km, gs)]));
				addMagz_d[indexDevice(i, j, k, gs)] += hxx*(dxp*(magnz_d[indexDevice(ip, j, k, gs)] - pz0) - dxm*(pz0 - magnz_d[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(magnz_d[indexDevice(i, jp, k, gs)] - pz0) - dym*(pz0 - magnz_d[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(magnz_d[indexDevice(i, j, kp, gs)] - pz0) - dzm*(pz0 - magnz_d[indexDevice(i, j, km, gs)]));
			}
		}
	}		
}

__global__ void NMR_2d_basic(char* actCell_d, double* diff_d,
	double *magnx_d, double* magny_d, double* magnz_d,
	double* mFx_d, double* mFy_d, double* mFz_d,
	double* nFx_d, double* nFy_d, double* nFz_d,
	double* velx_d, double* vely_d, double* velz_d,
	double* addMagx_d, double* addMagy_d, double* addMagz_d,
	double dTime, double dx, double dy, double dz, GridSize gs,
	int nVectorMult, int nVelocity, int nDiffusion, double* relT1_d, double* relT2_d, double equilibriumMagnetization)
{
	double hxx, hyy, hzz;
	int km, kp, jm, jp, im, ip;
	double tau1, tau2, dxm, dxp, dym, dyp, dzm, dzp;
	double px0, py0, pz0, anb, anx, any, anz, div, divx, divy, divz, pEq, pxEq, pyEq, pzEq;
	double b, bbxb, bbyb, bbzb;
	hxx = dTime / (dx*dx);
	hyy = dTime / (dy*dy);
	hzz = dTime / (dz*dz);

	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;

	//const int i = blockIdx.x * blockDim.x + threadIdx.x+ 1;
	//const int j = blockIdx.y * blockDim.y + threadIdx.y+ 1;
	//const int k = blockIdx.z * blockDim.z + threadIdx.z+ 1;
	//if ((i<gs.ik-1)&&(j<gs.jk-1)&&(k<gs.kk-1))

	const int gsni = gs.ik;
	const int gsnj = gs.jk;
	if (thIdx<gsni*gsnj)
	{
		const int i = (thIdx%gsni) + 1;
		const int j = (int)(thIdx / gsni) + 1;
		const int k = blockIdx.y + 1;

		km = k - 1;
		kp = k + 1;
		jm = j - 1;
		jp = j + 1;
		im = i - 1;
		ip = i + 1;
		if (actCell_d[indexDevice(i, j, k, gs)] == '1')
		{
			tau1 = 1.0 / relT1_d[indexDevice(i, j, k, gs)];//+
			tau2 = 1.0 / relT2_d[indexDevice(i, j, k, gs)];//+

			px0 = magnx_d[indexDevice(i, j, k, gs)];
			py0 = magny_d[indexDevice(i, j, k, gs)];
			pz0 = magnz_d[indexDevice(i, j, k, gs)];
			anb = sqrt(mFx_d[indexDevice(i, j, k, gs)] * mFx_d[indexDevice(i, j, k, gs)] + mFy_d[indexDevice(i, j, k, gs)] * mFy_d[indexDevice(i, j, k, gs)] + mFz_d[indexDevice(i, j, k, gs)] * mFz_d[indexDevice(i, j, k, gs)]);
			anx = mFx_d[indexDevice(i, j, k, gs)] / anb;
			any = mFy_d[indexDevice(i, j, k, gs)] / anb;
			anz = mFz_d[indexDevice(i, j, k, gs)] / anb;
			pEq = equilibriumMagnetization;
			pxEq = pEq*anx;
			pyEq = pEq*any;
			pzEq = pEq*anz;
			div = anx*px0 + any*py0 + anz*pz0;
			divx = anx*div;
			divy = any*div;
			divz = anz*div;

			b = dTime*4.78941714e+7;
			bbxb = b*nFx_d[indexDevice(i, j, k, gs)];
			bbyb = b*nFy_d[indexDevice(i, j, k, gs)];
			bbzb = b*nFz_d[indexDevice(i, j, k, gs)];
			//Solver itself
			addMagx_d[indexDevice(i, j, k, gs)] = px0 + dTime*tau1*(pxEq - divx) + dTime*tau2*(divx - px0);
			addMagy_d[indexDevice(i, j, k, gs)] = py0 + dTime*tau1*(pyEq - divy) + dTime*tau2*(divy - py0);
			addMagz_d[indexDevice(i, j, k, gs)] = pz0 + dTime*tau1*(pzEq - divz) + dTime*tau2*(divz - pz0);

			//Add smth with flags
			if (nVectorMult == 1)
			{
				addMagx_d[indexDevice(i, j, k, gs)] += 0.0 - bbyb*pz0 + bbzb*py0;
				addMagy_d[indexDevice(i, j, k, gs)] += 0.0 + bbxb*pz0 - bbzb*px0;
				addMagz_d[indexDevice(i, j, k, gs)] += 0.0 - bbxb*py0 + bbyb*px0;
			}
			if (nDiffusion == 1)
			{
				dxp = 2.0*(diff_d[indexDevice(ip, j, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(ip, j, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dxm = 2.0*(diff_d[indexDevice(im, j, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(im, j, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dyp = 2.0*(diff_d[indexDevice(i, jp, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, jp, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dym = 2.0*(diff_d[indexDevice(i, jm, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, jm, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dzp = 2.0*(diff_d[indexDevice(i, j, kp, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, j, kp, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dzm = 2.0*(diff_d[indexDevice(i, j, km, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, j, km, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				addMagx_d[indexDevice(i, j, k, gs)] += hxx*(dxp*(magnx_d[indexDevice(ip, j, k, gs)] - px0) - dxm*(px0 - magnx_d[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(magnx_d[indexDevice(i, jp, k, gs)] - px0) - dym*(px0 - magnx_d[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(magnx_d[indexDevice(i, j, kp, gs)] - px0) - dzm*(px0 - magnx_d[indexDevice(i, j, km, gs)]));
				addMagy_d[indexDevice(i, j, k, gs)] += hxx*(dxp*(magny_d[indexDevice(ip, j, k, gs)] - py0) - dxm*(py0 - magny_d[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(magny_d[indexDevice(i, jp, k, gs)] - py0) - dym*(py0 - magny_d[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(magny_d[indexDevice(i, j, kp, gs)] - py0) - dzm*(py0 - magny_d[indexDevice(i, j, km, gs)]));
				addMagz_d[indexDevice(i, j, k, gs)] += hxx*(dxp*(magnz_d[indexDevice(ip, j, k, gs)] - pz0) - dxm*(pz0 - magnz_d[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(magnz_d[indexDevice(i, jp, k, gs)] - pz0) - dym*(pz0 - magnz_d[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(magnz_d[indexDevice(i, j, kp, gs)] - pz0) - dzm*(pz0 - magnz_d[indexDevice(i, j, km, gs)]));
			}
		}
	}
}

__global__ void NMR_2d_predictor(char* actCell_d, 
	double *magnx_d, double* magny_d, double* magnz_d,
	double *predictorX, double* predictorY, double* predictorZ,
	double* velx_d, double* vely_d, double* velz_d,
	double dTime, double dx, double dy, double dz, GridSize gs)
{
	double hx, hy, hz;
	int km, kp, jm, jp, im, ip;
	double u, v, w, up, um, vp, vm, wp, wm;
	double px0, py0, pz0;
	hx = dTime / dx;
	hy = dTime / dy;
	hz = dTime / dz;

	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;

	const int gsni = gs.ik;
	const int gsnj = gs.jk;
	if (thIdx<gsni*gsnj)
	{
		const int i = (thIdx%gsni) + 1;
		const int j = (int)(thIdx / gsni) + 1;
		const int k = blockIdx.y + 1;

		km = k - 1;
		kp = k + 1;
		jm = j - 1;
		jp = j + 1;
		im = i - 1;
		ip = i + 1;
		if (actCell_d[indexDevice(i, j, k, gs)] == '1')
		{
			px0 = magnx_d[indexDevice(i, j, k, gs)];
			py0 = magny_d[indexDevice(i, j, k, gs)];
			pz0 = magnz_d[indexDevice(i, j, k, gs)];
			um = velx_d[indexDevice(i, j, k, gs)];
			vm = vely_d[indexDevice(i, j, k, gs)];
			wm = velz_d[indexDevice(i, j, k, gs)];
			up = velx_d[indexDevice(ip, j, k, gs)];
			vp = vely_d[indexDevice(i, jp, k, gs)];
			wp = velz_d[indexDevice(i, j, kp, gs)];
			u = 0.5*(um + up);
			v = 0.5*(vm + vp);
			w = 0.5*(wm + wp);
			if (actCell_d[indexDevice(im, j, k, gs)] == '0')
			{
				magnx_d[indexDevice(im, j, k, gs)] = magnx_d[indexDevice(i, j, k, gs)];
				magny_d[indexDevice(im, j, k, gs)] = magny_d[indexDevice(i, j, k, gs)];
				magnz_d[indexDevice(im, j, k, gs)] = magnz_d[indexDevice(i, j, k, gs)];
			}
			if (actCell_d[indexDevice(i, jm, k, gs)] == '0')
			{
				magnx_d[indexDevice(i, jm, k, gs)] = magnx_d[indexDevice(i, j, k, gs)];
				magny_d[indexDevice(i, jm, k, gs)] = magny_d[indexDevice(i, j, k, gs)];
				magnz_d[indexDevice(i, jm, k, gs)] = magnz_d[indexDevice(i, j, k, gs)];
			}
			if (actCell_d[indexDevice(i, j, km, gs)] == '0')
			{
				magnx_d[indexDevice(i, j, km, gs)] = magnx_d[indexDevice(i, j, k, gs)];
				magny_d[indexDevice(i, j, km, gs)] = magny_d[indexDevice(i, j, k, gs)];
				magnz_d[indexDevice(i, j, km, gs)] = magnz_d[indexDevice(i, j, k, gs)];
			}
			predictorX[indexDevice(i, j, k, gs)] = px0 -
				u*hx*(magnx_d[indexDevice(i, j, k, gs)] - magnx_d[indexDevice(im, j, k, gs)]) -
				v*hy*(magnx_d[indexDevice(i, j, k, gs)] - magnx_d[indexDevice(i, jm, k, gs)]) -
				w*hz*(magnx_d[indexDevice(i, j, k, gs)] - magnx_d[indexDevice(i, j, km, gs)]);
			predictorY[indexDevice(i, j, k, gs)] = py0 -
				u*hx*(magny_d[indexDevice(i, j, k, gs)] - magny_d[indexDevice(im, j, k, gs)]) -
				v*hy*(magny_d[indexDevice(i, j, k, gs)] - magny_d[indexDevice(i, jm, k, gs)]) -
				w*hz*(magny_d[indexDevice(i, j, k, gs)] - magny_d[indexDevice(i, j, km, gs)]);
			predictorZ[indexDevice(i, j, k, gs)] = pz0 -
				u*hx*(magnz_d[indexDevice(i, j, k, gs)] - magnz_d[indexDevice(im, j, k, gs)]) -
				v*hy*(magnz_d[indexDevice(i, j, k, gs)] - magnz_d[indexDevice(i, jm, k, gs)]) -
				w*hz*(magnz_d[indexDevice(i, j, k, gs)] - magnz_d[indexDevice(i, j, km, gs)]);
		}
	}
}

__global__ void NMR_2d_corrector(char* actCell_d,
	double *magnx_d, double* magny_d, double* magnz_d,
	double *predictorX, double* predictorY, double* predictorZ,
	double* correctorX, double* correctorY, double* correctorZ,
	double* velx_d, double* vely_d, double* velz_d,
	double dTime, double dx, double dy, double dz, GridSize gs)
{
	double hx, hy, hz;
	int kp, jp, ip;
	double u, v, w, up, um, vp, vm, wp, wm;
	double px0, py0, pz0;
	hx = dTime / dx;
	hy = dTime / dy;
	hz = dTime / dz;

	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;

	const int gsni = gs.ik;
	const int gsnj = gs.jk;
	if (thIdx<gsni*gsnj)
	{
		const int i = (thIdx%gsni) + 1;
		const int j = (int)(thIdx / gsni) + 1;
		const int k = blockIdx.y + 1;

		kp = k + 1;
		jp = j + 1;
		ip = i + 1;
		if (actCell_d[indexDevice(i, j, k, gs)] == '1')
		{
			px0 = magnx_d[indexDevice(i, j, k, gs)];
			py0 = magny_d[indexDevice(i, j, k, gs)];
			pz0 = magnz_d[indexDevice(i, j, k, gs)];
			um = velx_d[indexDevice(i, j, k, gs)];
			vm = vely_d[indexDevice(i, j, k, gs)];
			wm = velz_d[indexDevice(i, j, k, gs)];
			up = velx_d[indexDevice(ip, j, k, gs)];
			vp = vely_d[indexDevice(i, jp, k, gs)];
			wp = velz_d[indexDevice(i, j, kp, gs)];
			u = 0.5*(um + up);
			v = 0.5*(vm + vp);
			w = 0.5*(wm + wp);
			if (actCell_d[indexDevice(ip, j, k, gs)] == '0')
			{
				predictorX[indexDevice(ip, j, k, gs)] = predictorX[indexDevice(i, j, k, gs)];
				predictorY[indexDevice(ip, j, k, gs)] = predictorY[indexDevice(i, j, k, gs)];
				predictorZ[indexDevice(ip, j, k, gs)] = predictorZ[indexDevice(i, j, k, gs)];
			}
			if (actCell_d[indexDevice(i, jp, k, gs)] == '0')
			{
				predictorX[indexDevice(i, jp, k, gs)] = predictorX[indexDevice(i, j, k, gs)];
				predictorY[indexDevice(i, jp, k, gs)] = predictorY[indexDevice(i, j, k, gs)];
				predictorZ[indexDevice(i, jp, k, gs)] = predictorZ[indexDevice(i, j, k, gs)];
			}
			if (actCell_d[indexDevice(i, j, kp, gs)] == '0')
			{
				predictorX[indexDevice(i, j, kp, gs)] = predictorX[indexDevice(i, j, k, gs)];
				predictorY[indexDevice(i, j, kp, gs)] = predictorY[indexDevice(i, j, k, gs)];
				predictorZ[indexDevice(i, j, kp, gs)] = predictorZ[indexDevice(i, j, k, gs)];
			}
			correctorX[indexDevice(i, j, k, gs)] = 0.5*(px0 + predictorX[indexDevice(i, j, k, gs)] -
				u*hx*(predictorX[indexDevice(ip, j, k, gs)] - predictorX[indexDevice(i, j, k, gs)]) -
				v*hy*(predictorX[indexDevice(i, jp, k, gs)] - predictorX[indexDevice(i, j, k, gs)]) -
				w*hz*(predictorX[indexDevice(i, j, kp, gs)] - predictorX[indexDevice(i, j, k, gs)]));
			correctorY[indexDevice(i, j, k, gs)] = 0.5*(py0 + predictorY[indexDevice(i, j, k, gs)] -
				u*hx*(predictorY[indexDevice(ip, j, k, gs)] - predictorY[indexDevice(i, j, k, gs)]) -
				v*hy*(predictorY[indexDevice(i, jp, k, gs)] - predictorY[indexDevice(i, j, k, gs)]) -
				w*hz*(predictorY[indexDevice(i, j, kp, gs)] - predictorY[indexDevice(i, j, k, gs)]));
			correctorZ[indexDevice(i, j, k, gs)] = 0.5*(pz0 + predictorZ[indexDevice(i, j, k, gs)] -
				u*hx*(predictorZ[indexDevice(ip, j, k, gs)] - predictorZ[indexDevice(i, j, k, gs)]) -
				v*hy*(predictorZ[indexDevice(i, jp, k, gs)] - predictorZ[indexDevice(i, j, k, gs)]) -
				w*hz*(predictorZ[indexDevice(i, j, kp, gs)] - predictorZ[indexDevice(i, j, k, gs)]));
		}
	}
}

__global__ void NMR_2d_softner(char* actCell_d,
	double *magnx_d, double* magny_d, double* magnz_d,
	double* correctorX, double* correctorY, double* correctorZ,
	double* velx_d, double* vely_d, double* velz_d,
	double dTime, double dx, double dy, double dz, GridSize gs)
{
	int km, kp, jm, jp, im, ip, kmm, kpp, jmm, jpp, imm, ipp;
	double DmX, DpX, DmmX, DppX, DmY, DpY, DmmY, DppY, DmZ, DpZ, DmmZ, DppZ, q;
	double Qp, Qm;


	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;

	const int gsni = gs.ik-2;
	const int gsnj = gs.jk-2;
	if (thIdx<gsni*gsnj)
	{
		const int i = (thIdx%gsni) + 2;
		const int j = (int)(thIdx / gsni) + 2;
		const int k = blockIdx.y + 2;

		km = k - 1;
		kp = k + 1;
		jm = j - 1;
		jp = j + 1;
		im = i - 1;
		ip = i + 1;
		kmm = k - 2;
		kpp = k + 2;
		jmm = j - 2;
		jpp = j + 2;
		imm = i - 2;
		ipp = i + 2;

		if (actCell_d[indexDevice(i, j, k, gs)] == '1' && 
			actCell_d[indexDevice(im, j, k, gs)] == '1' && actCell_d[indexDevice(imm, j, k, gs)] == '1' &&
			actCell_d[indexDevice(ip, j, k, gs)] == '1' && actCell_d[indexDevice(ipp, j, k, gs)] == '1' &&
			actCell_d[indexDevice(i, jm, k, gs)] == '1' && actCell_d[indexDevice(i, jmm, k, gs)] == '1' &&
			actCell_d[indexDevice(i, jp, k, gs)] == '1' && actCell_d[indexDevice(i, jpp, k, gs)] == '1' &&
			actCell_d[indexDevice(i, j, km, gs)] == '1' && actCell_d[indexDevice(i, j, kmm, gs)] == '1' &&
			actCell_d[indexDevice(i, j, kp, gs)] == '1'&& actCell_d[indexDevice(i, j, kpp, gs)] == '1')
		{
			Qp = 0.0;
			Qm = 0.0;
			q = 0.1;
			if ((i > 1) && (i < gs.ik))
			{
				DmX = correctorX[indexDevice(i, j, k, gs)] - correctorX[indexDevice(im, j, k, gs)];
				DpX = correctorX[indexDevice(ip, j, k, gs)] - correctorX[indexDevice(i, j, k, gs)];
				DmmX = correctorX[indexDevice(im, j, k, gs)] - correctorX[indexDevice(imm, j, k, gs)];
				DppX = correctorX[indexDevice(ipp, j, k, gs)] - correctorX[indexDevice(ip, j, k, gs)];
				if (((DmX*DpX) <= 0.0) || ((DpX*DppX) <= 0.0)) Qp = DpX;
				if (((DmX*DpX) <= 0.0) || ((DmX*DmmX) <= 0.0)) Qm = DmX;
			}
			if (j > 1 && j < (gs.jk))
			{
				DmY = correctorX[indexDevice(i, j, k, gs)] - correctorX[indexDevice(i, jm, k, gs)];
				DpY = correctorX[indexDevice(i, jp, k, gs)] - correctorX[indexDevice(i, j, k, gs)];
				DmmY = correctorX[indexDevice(i, jm, k, gs)] - correctorX[indexDevice(i, jmm, k, gs)];
				DppY = correctorX[indexDevice(i, jpp, k, gs)] - correctorX[indexDevice(i, jp, k, gs)];
				if ((DmY*DpY <= 0.0) || (DpY*DppY <= 0.0)) Qp += DpY;
				if ((DmY*DpY <= 0.0) || (DmY*DmmY <= 0.0)) Qm += DmY;
			}
			if (k > 1 && k < (gs.kk))
			{
				DmZ = correctorX[indexDevice(i, j, k, gs)] - correctorX[indexDevice(i, j, km, gs)];
				DpZ = correctorX[indexDevice(i, j, kp, gs)] - correctorX[indexDevice(i, j, k, gs)];
				DmmZ = correctorX[indexDevice(i, j, km, gs)] - correctorX[indexDevice(i, j, kmm, gs)];
				DppZ = correctorX[indexDevice(i, j, kpp, gs)] - correctorX[indexDevice(i, j, kp, gs)];
				if ((DmZ*DpZ <= 0.0) || (DpZ*DppZ <= 0.0)) Qp += DpZ;
				if ((DmZ*DpZ <= 0.0) || (DmZ*DmmZ <= 0.0)) Qm += DmZ;
			}
			magnx_d[indexDevice(i, j, k, gs)] = correctorX[indexDevice(i, j, k, gs)] + q * (Qp - Qm);

			Qp = 0.0;
			Qm = 0.0;
			q = 0.1;
			if (i > 1 && i < (gs.ik))
			{
				DmX = correctorY[indexDevice(i, j, k, gs)] - correctorY[indexDevice(im, j, k, gs)];
				DpX = correctorY[indexDevice(ip, j, k, gs)] - correctorY[indexDevice(i, j, k, gs)];
				DmmX = correctorY[indexDevice(im, j, k, gs)] - correctorY[indexDevice(imm, j, k, gs)];
				DppX = correctorY[indexDevice(ipp, j, k, gs)] - correctorY[indexDevice(ip, j, k, gs)];
				if ((DmX*DpX <= 0.0) || (DpX*DppX <= 0.0)) Qp += DpX;
				if ((DmX*DpX <= 0.0) || (DmX*DmmX <= 0.0)) Qm += DmX;
			}
			if (j > 1 && j < (gs.jk))
			{
				DmY = correctorY[indexDevice(i, j, k, gs)] - correctorY[indexDevice(i, jm, k, gs)];
				DpY = correctorY[indexDevice(i, jp, k, gs)] - correctorY[indexDevice(i, j, k, gs)];
				DmmY = correctorY[indexDevice(i, jm, k, gs)] - correctorY[indexDevice(i, jmm, k, gs)];
				DppY = correctorY[indexDevice(i, jpp, k, gs)] - correctorY[indexDevice(i, jp, k, gs)];
				if ((DmY*DpY <= 0.0) || (DpY*DppY <= 0.0)) Qp += DpY;
				if ((DmY*DpY <= 0.0) || (DmY*DmmY <= 0.0)) Qm += DmY;
			}
			if (k > 1 && k < (gs.kk))
			{
				DmZ = correctorY[indexDevice(i, j, k, gs)] - correctorY[indexDevice(i, j, km, gs)];
				DpZ = correctorY[indexDevice(i, j, kp, gs)] - correctorY[indexDevice(i, j, k, gs)];
				DmmZ = correctorY[indexDevice(i, j, km, gs)] - correctorY[indexDevice(i, j, kmm, gs)];
				DppZ = correctorY[indexDevice(i, j, kpp, gs)] - correctorY[indexDevice(i, j, kp, gs)];
				if ((DmZ*DpZ <= 0.0) || (DpZ*DppZ <= 0.0)) Qp += DpZ;
				if ((DmZ*DpZ <= 0.0) || (DmZ*DmmZ <= 0.0)) Qm += DmZ;
			}
			magny_d[indexDevice(i, j, k, gs)] = correctorY[indexDevice(i, j, k, gs)] + q * (Qp - Qm);

			Qp = 0.0;
			Qm = 0.0;
			q = 0.1;
			if (i > 1 && i < (gs.ik))
			{
				DmX = correctorZ[indexDevice(i, j, k, gs)] - correctorZ[indexDevice(im, j, k, gs)];
				DpX = correctorZ[indexDevice(ip, j, k, gs)] - correctorZ[indexDevice(i, j, k, gs)];
				DmmX = correctorZ[indexDevice(im, j, k, gs)] - correctorZ[indexDevice(imm, j, k, gs)];
				DppX = correctorZ[indexDevice(ipp, j, k, gs)] - correctorZ[indexDevice(ip, j, k, gs)];
				if ((DmX*DpX <= 0.0) || (DpX*DppX <= 0.0)) Qp += DpX;
				if ((DmX*DpX <= 0.0) || (DmX*DmmX <= 0.0)) Qm += DmX;
			}
			if (j > 1 && j < (gs.jk))
			{
				DmY = correctorZ[indexDevice(i, j, k, gs)] - correctorZ[indexDevice(i, jm, k, gs)];
				DpY = correctorZ[indexDevice(i, jp, k, gs)] - correctorZ[indexDevice(i, j, k, gs)];
				DmmY = correctorZ[indexDevice(i, jm, k, gs)] - correctorZ[indexDevice(i, jmm, k, gs)];
				DppY = correctorZ[indexDevice(i, jpp, k, gs)] - correctorZ[indexDevice(i, jp, k, gs)];
				if ((DmY*DpY <= 0.0) || (DpY*DppY <= 0.0)) Qp += DpY;
				if ((DmY*DpY <= 0.0) || (DmY*DmmY <= 0.0)) Qm += DmY;
			}
			if (k > 1 && k < (gs.kk))
			{
				DmZ = correctorZ[indexDevice(i, j, k, gs)] - correctorZ[indexDevice(i, j, km, gs)];
				DpZ = correctorZ[indexDevice(i, j, kp, gs)] - correctorZ[indexDevice(i, j, k, gs)];
				DmmZ = correctorZ[indexDevice(i, j, km, gs)] - correctorZ[indexDevice(i, j, kmm, gs)];
				DppZ = correctorZ[indexDevice(i, j, kpp, gs)] - correctorZ[indexDevice(i, j, kp, gs)];
				if ((DmZ*DpZ <= 0.0) || (DpZ*DppZ <= 0.0)) Qp += DpZ;
				if ((DmZ*DpZ <= 0.0) || (DmZ*DmmZ <= 0.0)) Qm += DmZ;
			}
			magnz_d[indexDevice(i, j, k, gs)] = correctorZ[indexDevice(i, j, k, gs)] + q * (Qp - Qm);
		}
		else
		{
			magnx_d[indexDevice(i, j, k, gs)] = correctorX[indexDevice(i, j, k, gs)];
			magny_d[indexDevice(i, j, k, gs)] = correctorY[indexDevice(i, j, k, gs)];
			magnz_d[indexDevice(i, j, k, gs)] = correctorZ[indexDevice(i, j, k, gs)];
		}
	//magnx_d[indexDevice(i, j, k, gs)] = correctorX[indexDevice(i, j, k, gs)];
	//magny_d[indexDevice(i, j, k, gs)] = correctorY[indexDevice(i, j, k, gs)];
	//magnz_d[indexDevice(i, j, k, gs)] = correctorZ[indexDevice(i, j, k, gs)];
	}
}
// Upgade Mackormack

__global__ void NMR_predictor_upgrade(char* actCell_d, double* diff_d,
	double *magnx_d, double* magny_d, double* magnz_d,
	double* mFx_d, double* mFy_d, double* mFz_d,
	double* nFx_d, double* nFy_d, double* nFz_d,
	double *predictorX, double* predictorY, double* predictorZ,
	double* correctorX, double* correctorY, double* correctorZ,
	double* velx_d, double* vely_d, double* velz_d,
	double dTime, double dx, double dy, double dz, GridSize gs,
	int nVectorMult, int nVelocity, int nDiffusion, double* relT1_d, double* relT2_d, double equilibriumMagnetization)
{


	double hx, hy, hz;
	int km, kp, jm, jp, im, ip;
	double u, v, w, up, um, vp, vm, wp, wm;
	double px0, py0, pz0;
	double hxx, hyy, hzz;
	double tau1, tau2, dxm, dxp, dym, dyp, dzm, dzp;
	double anb, anx, any, anz, div, divx, divy, divz, pEq, pxEq, pyEq, pzEq;
	double b, bbxb, bbyb, bbzb;
	double relPartX, relPartY, relPartZ;
	double vectorMultX, vectorMultY, vectorMultZ;
	double diffPartX, diffPartY, diffPartZ;
	double velPartX, velPartY, velPartZ;
	hx = dTime / dx;
	hy = dTime / dy;
	hz = dTime / dz;
	hxx = dTime / (dx*dx);
	hyy = dTime / (dy*dy);
	hzz = dTime / (dz*dz);

	relPartX = 0.0;
	relPartY = 0.0;
	relPartZ = 0.0;
	vectorMultX = 0.0;
	vectorMultY = 0.0;
	vectorMultZ = 0.0;
	diffPartX = 0.0;
	diffPartY = 0.0;
	diffPartZ = 0.0;
	velPartX = 0.0;
	velPartY = 0.0;
	velPartZ = 0.0;
	

	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;

	const int gsni = gs.ik;
	const int gsnj = gs.jk;
	if (thIdx<gsni*gsnj)
	{
		const int i = (thIdx%gsni) + 1;
		const int j = (int)(thIdx / gsni) + 1;
		const int k = blockIdx.y + 1;

		km = k - 1;
		kp = k + 1;
		jm = j - 1;
		jp = j + 1;
		im = i - 1;
		ip = i + 1;
		if (actCell_d[indexDevice(i, j, k, gs)] == '1')
		{
			px0 = magnx_d[indexDevice(i, j, k, gs)];
			py0 = magny_d[indexDevice(i, j, k, gs)];
			pz0 = magnz_d[indexDevice(i, j, k, gs)];
			tau1 = 1.0 / relT1_d[indexDevice(i, j, k, gs)];
			tau2 = 1.0 / relT2_d[indexDevice(i, j, k, gs)];
			anb = sqrt(mFx_d[indexDevice(i, j, k, gs)] * mFx_d[indexDevice(i, j, k, gs)] + mFy_d[indexDevice(i, j, k, gs)] * mFy_d[indexDevice(i, j, k, gs)] + mFz_d[indexDevice(i, j, k, gs)] * mFz_d[indexDevice(i, j, k, gs)]);
			anx = mFx_d[indexDevice(i, j, k, gs)] / anb;
			any = mFy_d[indexDevice(i, j, k, gs)] / anb;
			anz = mFz_d[indexDevice(i, j, k, gs)] / anb;
			pEq = equilibriumMagnetization;
			pxEq = pEq*anx;
			pyEq = pEq*any;
			pzEq = pEq*anz;
			div = anx*px0 + any*py0 + anz*pz0;
			divx = anx*div;
			divy = any*div;
			divz = anz*div;

			b = dTime*4.78941714e+7;
			bbxb = b*nFx_d[indexDevice(i, j, k, gs)];
			bbyb = b*nFy_d[indexDevice(i, j, k, gs)];
			bbzb = b*nFz_d[indexDevice(i, j, k, gs)];
			//Solver itself
			relPartX = dTime*tau1*(pxEq - divx) + dTime*tau2*(divx - px0);
			relPartY = dTime*tau1*(pyEq - divy) + dTime*tau2*(divy - py0);
			relPartZ = dTime*tau1*(pzEq - divz) + dTime*tau2*(divz - pz0);

			//Add smth with flags
			if (nVectorMult == 1)
			{
				vectorMultX = 0.0 - bbyb*pz0 + bbzb*py0;
				vectorMultY = 0.0 + bbxb*pz0 - bbzb*px0;
				vectorMultZ = 0.0 - bbxb*py0 + bbyb*px0;
			}

			if (nDiffusion == 1)
			{
				dxp = 2.0*(diff_d[indexDevice(ip, j, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(ip, j, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dxm = 2.0*(diff_d[indexDevice(im, j, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(im, j, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dyp = 2.0*(diff_d[indexDevice(i, jp, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, jp, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dym = 2.0*(diff_d[indexDevice(i, jm, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, jm, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dzp = 2.0*(diff_d[indexDevice(i, j, kp, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, j, kp, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dzm = 2.0*(diff_d[indexDevice(i, j, km, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, j, km, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				diffPartX = hxx*(dxp*(magnx_d[indexDevice(ip, j, k, gs)] - px0) - dxm*(px0 - magnx_d[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(magnx_d[indexDevice(i, jp, k, gs)] - px0) - dym*(px0 - magnx_d[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(magnx_d[indexDevice(i, j, kp, gs)] - px0) - dzm*(px0 - magnx_d[indexDevice(i, j, km, gs)]));
				diffPartY = hxx*(dxp*(magny_d[indexDevice(ip, j, k, gs)] - py0) - dxm*(py0 - magny_d[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(magny_d[indexDevice(i, jp, k, gs)] - py0) - dym*(py0 - magny_d[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(magny_d[indexDevice(i, j, kp, gs)] - py0) - dzm*(py0 - magny_d[indexDevice(i, j, km, gs)]));
				diffPartZ = hxx*(dxp*(magnz_d[indexDevice(ip, j, k, gs)] - pz0) - dxm*(pz0 - magnz_d[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(magnz_d[indexDevice(i, jp, k, gs)] - pz0) - dym*(pz0 - magnz_d[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(magnz_d[indexDevice(i, j, kp, gs)] - pz0) - dzm*(pz0 - magnz_d[indexDevice(i, j, km, gs)]));
			}

			if (nVelocity == 1)
			{
				um = velx_d[indexDevice(i, j, k, gs)];
				vm = vely_d[indexDevice(i, j, k, gs)];
				wm = velz_d[indexDevice(i, j, k, gs)];
				up = velx_d[indexDevice(ip, j, k, gs)];
				vp = vely_d[indexDevice(i, jp, k, gs)];
				wp = velz_d[indexDevice(i, j, kp, gs)];
				u = 0.5*(um + up);
				v = 0.5*(vm + vp);
				w = 0.5*(wm + wp);
				if (actCell_d[indexDevice(im, j, k, gs)] == '0')
				{
					magnx_d[indexDevice(im, j, k, gs)] = magnx_d[indexDevice(i, j, k, gs)];
					magny_d[indexDevice(im, j, k, gs)] = magny_d[indexDevice(i, j, k, gs)];
					magnz_d[indexDevice(im, j, k, gs)] = magnz_d[indexDevice(i, j, k, gs)];
				}
				if (actCell_d[indexDevice(i, jm, k, gs)] == '0')
				{
					magnx_d[indexDevice(i, jm, k, gs)] = magnx_d[indexDevice(i, j, k, gs)];
					magny_d[indexDevice(i, jm, k, gs)] = magny_d[indexDevice(i, j, k, gs)];
					magnz_d[indexDevice(i, jm, k, gs)] = magnz_d[indexDevice(i, j, k, gs)];
				}
				if (actCell_d[indexDevice(i, j, km, gs)] == '0')
				{
					magnx_d[indexDevice(i, j, km, gs)] = magnx_d[indexDevice(i, j, k, gs)];
					magny_d[indexDevice(i, j, km, gs)] = magny_d[indexDevice(i, j, k, gs)];
					magnz_d[indexDevice(i, j, km, gs)] = magnz_d[indexDevice(i, j, k, gs)];
				}
				velPartX = 0.0 -
					u*hx*(magnx_d[indexDevice(i, j, k, gs)] - magnx_d[indexDevice(im, j, k, gs)]) -
					v*hy*(magnx_d[indexDevice(i, j, k, gs)] - magnx_d[indexDevice(i, jm, k, gs)]) -
					w*hz*(magnx_d[indexDevice(i, j, k, gs)] - magnx_d[indexDevice(i, j, km, gs)]);
				velPartY = 0.0 -
					u*hx*(magny_d[indexDevice(i, j, k, gs)] - magny_d[indexDevice(im, j, k, gs)]) -
					v*hy*(magny_d[indexDevice(i, j, k, gs)] - magny_d[indexDevice(i, jm, k, gs)]) -
					w*hz*(magny_d[indexDevice(i, j, k, gs)] - magny_d[indexDevice(i, j, km, gs)]);
				velPartZ = 0.0 -
					u*hx*(magnz_d[indexDevice(i, j, k, gs)] - magnz_d[indexDevice(im, j, k, gs)]) -
					v*hy*(magnz_d[indexDevice(i, j, k, gs)] - magnz_d[indexDevice(i, jm, k, gs)]) -
					w*hz*(magnz_d[indexDevice(i, j, k, gs)] - magnz_d[indexDevice(i, j, km, gs)]);
			}
			predictorX[indexDevice(i, j, k, gs)] = px0 + relPartX + vectorMultX + diffPartX + velPartX;
			predictorY[indexDevice(i, j, k, gs)] = py0 + relPartY + vectorMultY + diffPartY + velPartY;
			predictorZ[indexDevice(i, j, k, gs)] = pz0 + relPartZ + vectorMultZ + diffPartZ + velPartZ;
		}
	}
}

__global__ void NMR_corrector_upgrade(char* actCell_d, double* diff_d,
	double *magnx_d, double* magny_d, double* magnz_d,
	double* mFx_d, double* mFy_d, double* mFz_d,
	double* nFx_d, double* nFy_d, double* nFz_d,
	double *predictorX, double* predictorY, double* predictorZ,
	double* correctorX, double* correctorY, double* correctorZ,
	double* velx_d, double* vely_d, double* velz_d,
	double dTime, double dx, double dy, double dz, GridSize gs,
	int nVectorMult, int nVelocity, int nDiffusion, double* relT1_d, double* relT2_d, double equilibriumMagnetization)
{


	double hx, hy, hz;
	int km, kp, jm, jp, im, ip;
	double u, v, w, up, um, vp, vm, wp, wm;
	double px0, py0, pz0;
	double hxx, hyy, hzz;
	double tau1, tau2, dxm, dxp, dym, dyp, dzm, dzp;
	double anb, anx, any, anz, div, divx, divy, divz, pEq, pxEq, pyEq, pzEq;
	double b, bbxb, bbyb, bbzb;
	double relPartX, relPartY, relPartZ;
	double vectorMultX, vectorMultY, vectorMultZ;
	double diffPartX, diffPartY, diffPartZ;
	double velPartX, velPartY, velPartZ;
	hx = dTime / dx;
	hy = dTime / dy;
	hz = dTime / dz;
	hxx = dTime / (dx*dx);
	hyy = dTime / (dy*dy);
	hzz = dTime / (dz*dz);

	relPartX = 0.0;
	relPartY = 0.0;
	relPartZ = 0.0;
	vectorMultX = 0.0;
	vectorMultY = 0.0;
	vectorMultZ = 0.0;
	diffPartX = 0.0;
	diffPartY = 0.0;
	diffPartZ = 0.0;
	velPartX = 0.0;
	velPartY = 0.0;
	velPartZ = 0.0;


	int thIdx = blockIdx.x * blockDim.x + threadIdx.x;

	const int gsni = gs.ik;
	const int gsnj = gs.jk;
	if (thIdx<gsni*gsnj)
	{
		const int i = (thIdx%gsni) + 1;
		const int j = (int)(thIdx / gsni) + 1;
		const int k = blockIdx.y + 1;

		km = k - 1;
		kp = k + 1;
		jm = j - 1;
		jp = j + 1;
		im = i - 1;
		ip = i + 1;
		if (actCell_d[indexDevice(i, j, k, gs)] == '1')
		{
			px0 = predictorX[indexDevice(i, j, k, gs)];
			py0 = predictorY[indexDevice(i, j, k, gs)];
			pz0 = predictorZ[indexDevice(i, j, k, gs)];
			tau1 = 1.0 / relT1_d[indexDevice(i, j, k, gs)];
			tau2 = 1.0 / relT2_d[indexDevice(i, j, k, gs)];
			anb = sqrt(mFx_d[indexDevice(i, j, k, gs)] * mFx_d[indexDevice(i, j, k, gs)] + mFy_d[indexDevice(i, j, k, gs)] * mFy_d[indexDevice(i, j, k, gs)] + mFz_d[indexDevice(i, j, k, gs)] * mFz_d[indexDevice(i, j, k, gs)]);
			anx = mFx_d[indexDevice(i, j, k, gs)] / anb;
			any = mFy_d[indexDevice(i, j, k, gs)] / anb;
			anz = mFz_d[indexDevice(i, j, k, gs)] / anb;
			pEq = equilibriumMagnetization;
			pxEq = pEq*anx;
			pyEq = pEq*any;
			pzEq = pEq*anz;
			div = anx*px0 + any*py0 + anz*pz0;
			divx = anx*div;
			divy = any*div;
			divz = anz*div;

			b = dTime*4.78941714e+7;
			bbxb = b*nFx_d[indexDevice(i, j, k, gs)];
			bbyb = b*nFy_d[indexDevice(i, j, k, gs)];
			bbzb = b*nFz_d[indexDevice(i, j, k, gs)];
			//Solver itself
			relPartX = dTime*tau1*(pxEq - divx) + dTime*tau2*(divx - px0);
			relPartY = dTime*tau1*(pyEq - divy) + dTime*tau2*(divy - py0);
			relPartZ = dTime*tau1*(pzEq - divz) + dTime*tau2*(divz - pz0);

			//Add smth with flags
			if (nVectorMult == 1)
			{
				vectorMultX = 0.0 - bbyb*pz0 + bbzb*py0;
				vectorMultY = 0.0 + bbxb*pz0 - bbzb*px0;
				vectorMultZ = 0.0 - bbxb*py0 + bbyb*px0;
			}

			if (nDiffusion == 1)
			{
				dxp = 2.0*(diff_d[indexDevice(ip, j, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(ip, j, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dxm = 2.0*(diff_d[indexDevice(im, j, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(im, j, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dyp = 2.0*(diff_d[indexDevice(i, jp, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, jp, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dym = 2.0*(diff_d[indexDevice(i, jm, k, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, jm, k, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dzp = 2.0*(diff_d[indexDevice(i, j, kp, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, j, kp, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				dzm = 2.0*(diff_d[indexDevice(i, j, km, gs)] * diff_d[indexDevice(i, j, k, gs)]) / (diff_d[indexDevice(i, j, km, gs)] + diff_d[indexDevice(i, j, k, gs)] + 1.0e-25);
				diffPartX = hxx*(dxp*(predictorX[indexDevice(ip, j, k, gs)] - px0) - dxm*(px0 - predictorX[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(predictorX[indexDevice(i, jp, k, gs)] - px0) - dym*(px0 - predictorX[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(predictorX[indexDevice(i, j, kp, gs)] - px0) - dzm*(px0 - predictorX[indexDevice(i, j, km, gs)]));
				diffPartY = hxx*(dxp*(predictorY[indexDevice(ip, j, k, gs)] - py0) - dxm*(py0 - predictorY[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(predictorY[indexDevice(i, jp, k, gs)] - py0) - dym*(py0 - predictorY[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(predictorY[indexDevice(i, j, kp, gs)] - py0) - dzm*(py0 - predictorY[indexDevice(i, j, km, gs)]));
				diffPartZ = hxx*(dxp*(predictorZ[indexDevice(ip, j, k, gs)] - pz0) - dxm*(pz0 - predictorZ[indexDevice(im, j, k, gs)]))
					+ hyy*(dyp*(predictorZ[indexDevice(i, jp, k, gs)] - pz0) - dym*(pz0 - predictorZ[indexDevice(i, jm, k, gs)]))
					+ hzz*(dzp*(predictorZ[indexDevice(i, j, kp, gs)] - pz0) - dzm*(pz0 - predictorZ[indexDevice(i, j, km, gs)]));
			}

			if (nVelocity == 1)
			{
				um = velx_d[indexDevice(i, j, k, gs)];
				vm = vely_d[indexDevice(i, j, k, gs)];
				wm = velz_d[indexDevice(i, j, k, gs)];
				up = velx_d[indexDevice(ip, j, k, gs)];
				vp = vely_d[indexDevice(i, jp, k, gs)];
				wp = velz_d[indexDevice(i, j, kp, gs)];
				u = 0.5*(um + up);
				v = 0.5*(vm + vp);
				w = 0.5*(wm + wp);
				if (actCell_d[indexDevice(ip, j, k, gs)] == '0')
				{
					predictorX[indexDevice(ip, j, k, gs)] = predictorX[indexDevice(i, j, k, gs)];
					predictorY[indexDevice(ip, j, k, gs)] = predictorY[indexDevice(i, j, k, gs)];
					predictorZ[indexDevice(ip, j, k, gs)] = predictorZ[indexDevice(i, j, k, gs)];
				}
				if (actCell_d[indexDevice(i, jp, k, gs)] == '0')
				{
					predictorX[indexDevice(i, jp, k, gs)] = predictorX[indexDevice(i, j, k, gs)];
					predictorY[indexDevice(i, jp, k, gs)] = predictorY[indexDevice(i, j, k, gs)];
					predictorZ[indexDevice(i, jp, k, gs)] = predictorZ[indexDevice(i, j, k, gs)];
				}
				if (actCell_d[indexDevice(i, j, kp, gs)] == '0')
				{
					predictorX[indexDevice(i, j, kp, gs)] = predictorX[indexDevice(i, j, k, gs)];
					predictorY[indexDevice(i, j, kp, gs)] = predictorY[indexDevice(i, j, k, gs)];
					predictorZ[indexDevice(i, j, kp, gs)] = predictorZ[indexDevice(i, j, k, gs)];
				}
				velPartX = 0.0 -
					u*hx*(predictorX[indexDevice(ip, j, k, gs)] - predictorX[indexDevice(i, j, k, gs)]) -
					v*hy*(predictorX[indexDevice(i, jp, k, gs)] - predictorX[indexDevice(i, j, k, gs)]) -
					w*hz*(predictorX[indexDevice(i, j, kp, gs)] - predictorX[indexDevice(i, j, k, gs)]);
				velPartY = 0.0 -
					u*hx*(predictorY[indexDevice(ip, j, k, gs)] - predictorY[indexDevice(i, j, k, gs)]) -
					v*hy*(predictorY[indexDevice(i, jp, k, gs)] - predictorY[indexDevice(i, j, k, gs)]) -
					w*hz*(predictorY[indexDevice(i, j, kp, gs)] - predictorY[indexDevice(i, j, k, gs)]);
				velPartZ = 0.0 -
					u*hx*(predictorZ[indexDevice(ip, j, k, gs)] - predictorZ[indexDevice(i, j, k, gs)]) -
					v*hy*(predictorZ[indexDevice(i, jp, k, gs)] - predictorZ[indexDevice(i, j, k, gs)]) -
					w*hz*(predictorZ[indexDevice(i, j, kp, gs)] - predictorZ[indexDevice(i, j, k, gs)]);
			}
			correctorX[indexDevice(i, j, k, gs)] = 0.5*(magnx_d[indexDevice(i, j, k, gs)] + predictorX[indexDevice(i, j, k, gs)] + relPartX + vectorMultX + diffPartX + velPartX);
			correctorY[indexDevice(i, j, k, gs)] = 0.5*(magny_d[indexDevice(i, j, k, gs)] + predictorY[indexDevice(i, j, k, gs)] + relPartY + vectorMultY + diffPartY + velPartY);
			correctorZ[indexDevice(i, j, k, gs)] = 0.5*(magnz_d[indexDevice(i, j, k, gs)] + predictorZ[indexDevice(i, j, k, gs)] + relPartZ + vectorMultZ + diffPartZ + velPartZ);
		}
	}
}

__global__ void BC_ZeroDer_X(double* F_d, GridSize gs)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	if ((j<(gs.jk + 2)) && (k<(gs.kk + 2)))
	{
		F_d[indexDevice(0, j, k, gs)] = F_d[indexDevice(1, j, k, gs)];
		F_d[indexDevice(gs.ik + 1, j, k, gs)] = F_d[indexDevice(gs.ik, j, k, gs)];
	}
}

__global__ void BC_ZeroDer_Y(double* F_d, GridSize gs)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	if((i<(gs.ik+2))&&(k<(gs.kk+2))) 
	{
		F_d[indexDevice(i, 0, k, gs)] = F_d[indexDevice(i, 1, k, gs)];
		F_d[indexDevice(i, gs.jk + 1, k, gs)] = F_d[indexDevice(i, gs.jk, k, gs)];
	}	
}

__global__ void BC_ZeroDer_Z(double* F_d, GridSize gs)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if((i<(gs.ik+2))&&(j<(gs.jk+2))) 
	{
		F_d[indexDevice(i, j, 0, gs)] = F_d[indexDevice(i, j, 1, gs)];//2 * F_d[indexDevice(i, j, 1, gs)] - F_d[indexDevice(i, j, 2, gs)];
		F_d[indexDevice(i, j, gs.kk + 1, gs)] = F_d[indexDevice(i, j, gs.kk, gs)];
	}	
}

__global__ void BC_Inf_X(double* F_d, GridSize gs)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	if ((j<(gs.jk + 2)) && (k<(gs.kk + 2)))
	{
		F_d[indexDevice(0, j, k, gs)] = F_d[indexDevice(gs.ik, j, k, gs)];
		F_d[indexDevice(gs.ik + 1, j, k, gs)] = F_d[indexDevice(1, j, k, gs)]; 
	}
}

__global__ void BC_Inf_Y(double* F_d, GridSize gs)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i<(gs.ik + 2)) && (k<(gs.kk + 2)))
	{
		F_d[indexDevice(i, 0, k, gs)] = F_d[indexDevice(i, gs.jk, k, gs)]; 
		F_d[indexDevice(i, gs.jk + 1, k, gs)] = F_d[indexDevice(i, 1, k, gs)];
	}
}

__global__ void BC_Inf_Z(double* F_d, GridSize gs)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i<(gs.ik + 2)) && (j<(gs.jk + 2)))
	{
		F_d[indexDevice(i, j, 0, gs)] = F_d[indexDevice(i, j, gs.kk, gs)]; 
		F_d[indexDevice(i, j, gs.kk + 1, gs)] = F_d[indexDevice(i, j, 1, gs)];//2 * F_d[indexDevice(i, j, 1, gs)] - F_d[indexDevice(i, j, 2, gs)];
	}
}

__global__ void BC_Advanced_XM(double* F_d, double Val, GridSize gs)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	if ((j<(gs.jk + 2)) && (k<(gs.kk + 2)))
	{
		F_d[indexDevice(0, j, k, gs)] = Val;
	}
}

__global__ void BC_Advanced_XP(double* F_d, double Val, GridSize gs)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	if ((j<(gs.jk + 2)) && (k<(gs.kk + 2)))
	{
		F_d[indexDevice(gs.ik + 1, j, k, gs)] = Val;
	}
}


void NmrRelaxationSolve::solve_gpu()
{
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	const int num_el = (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2);
	int lastNum;
	/*CUDA_arrays declaration*/
	//char* actCell_d;
	//double* diff_d, *magnx_d, *magny_d, *magnz_d, 
	//	*mFx_d, *mFy_d, *mFz_d, *nFx_d, *nFy_d, *nFz_d, 
	//	*velx_d, *vely_d, *velz_d, 
	//	*addMagx_d, *addMagy_d, *addMagz_d,
	//	*predictorX, *predictorY, *predictorZ,
	//	*correctorX, *correctorY, *correctorZ,
	//	*softnerX, *softnerY, *softnerZ,
	//	*relT1_d,* relT2_d;
	InitializeGpu();
	BoundaryConditionsForVectorField_gpu(magnx_d, magny_d, magnz_d, &magnetization, gs);
	char buf[256], buf_f[256];

	/*CUDA_arrays initialization*/
//	gpuErrchk(cudaMalloc((void **)&actCell_d, num_el*sizeof(char)));
//
//	gpuErrchk(cudaMalloc((void **)&diff_d, num_el*sizeof(double)));
//
//	gpuErrchk(cudaMalloc((void **)&magnx_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&magny_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&magnz_d, num_el*sizeof(double)));
//
//	gpuErrchk(cudaMalloc((void **)&mFx_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&mFy_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&mFz_d, num_el*sizeof(double)));
//
//	gpuErrchk(cudaMalloc((void **)&nFx_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&nFy_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&nFz_d, num_el*sizeof(double)));
//
//	gpuErrchk(cudaMalloc((void **)&velx_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&vely_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&velz_d, num_el*sizeof(double)));
//
//	gpuErrchk(cudaMalloc((void **)&addMagx_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&addMagy_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&addMagz_d, num_el*sizeof(double)));
//
//	gpuErrchk(cudaMalloc((void **)&predictorX, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&predictorY, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&predictorZ, num_el*sizeof(double)));
//
//	gpuErrchk(cudaMalloc((void **)&correctorX, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&correctorY, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&correctorZ, num_el*sizeof(double)));
//
//	gpuErrchk(cudaMalloc((void **)&softnerX, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&softnerY, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&softnerZ, num_el*sizeof(double)));
//
//	gpuErrchk(cudaMalloc((void **)&relT1_d, num_el*sizeof(double)));
//	gpuErrchk(cudaMalloc((void **)&relT2_d, num_el*sizeof(double)));
////
//	cudaMemset(predictorX, 0.0, num_el*sizeof(double));
//	cudaMemset(predictorY, 0.0, num_el*sizeof(double));
//	cudaMemset(predictorZ, 0.0, num_el*sizeof(double));
//	cudaMemset(correctorX, 0.0, num_el*sizeof(double));
//	cudaMemset(correctorY, 0.0, num_el*sizeof(double));
//	cudaMemset(correctorZ, 0.0, num_el*sizeof(double));
//	
//	/*CUDA_arrays set*/
//	gpuErrchk(cudaMemcpy(actCell_d, actCell, num_el*sizeof(char), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(diff_d, diffusion, num_el*sizeof(double), cudaMemcpyHostToDevice));
//
//	gpuErrchk(cudaMemcpy(magnx_d, magnetization.x, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(magny_d, magnetization.y, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(magnz_d, magnetization.z, num_el*sizeof(double), cudaMemcpyHostToDevice));
//
//	gpuErrchk(cudaMemcpy(mFx_d, mainField.x, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(mFy_d, mainField.y, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(mFz_d, mainField.z, num_el*sizeof(double), cudaMemcpyHostToDevice));
//
//	gpuErrchk(cudaMemcpy(nFx_d, noiseField.x, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(nFy_d, noiseField.y, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(nFz_d, noiseField.z, num_el*sizeof(double), cudaMemcpyHostToDevice));
//
//	gpuErrchk(cudaMemcpy(velx_d, velocity.x, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(vely_d, velocity.y, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(velz_d, velocity.z, num_el*sizeof(double), cudaMemcpyHostToDevice));
//
//	gpuErrchk(cudaMemcpy(addMagx_d, addMagnetization.x, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(addMagy_d, addMagnetization.y, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(addMagz_d, addMagnetization.z, num_el*sizeof(double), cudaMemcpyHostToDevice));
//
//	gpuErrchk(cudaMemcpy(relT1_d, relTime.t1, num_el*sizeof(double), cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(relT2_d, relTime.t2, num_el*sizeof(double), cudaMemcpyHostToDevice));
	/*CUDA_GRID_PARAMETERS*/
	//Cuda kernel launch parameters
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


	dim3 dimBlock = dim3(CUDA_THREAD_PER_BLOCK);
	dim3 dimGrid2d = dim3((int)(ceil((gs.ik)*(double)(gs.jk) / CUDA_THREAD_PER_BLOCK)), gs.kk);
	dim3 dimGrid2dM1 = dim3((int)(ceil((gs.ik-2)*(double)(gs.jk-2) / CUDA_THREAD_PER_BLOCK)), (gs.kk-2));
	
	ofstream res("results_gpu.csv");
	MPI_Barrier(MPI_COMM_WORLD);
	if (MPI_rank == 0) res << "Time; SumX; SumY; SumZ" << endl;

	//nStep = 1;
	for (int i = 0; i < nStep + 1; i++)
	{
		currentTime = i*dTime;
		ExplicitMakUpgrade();
		Sequence_gpu(magnx_d, magny_d, magnz_d);
		BoundaryConditionsForVectorField_gpu(magnx_d, magny_d, magnz_d, &magnetization, gs);
		if ((i % ndPrint) == 0)
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));

			OutputX(&magnetization);
			OutputY(&magnetization);
			OutputZ(&magnetization);
			lastNum = i;
			sprintf(buf_f, "fields/field_%d_%d_%d_%d", PulseNum[0], PulseNum[1], PulseNum[2], i);
			//WriteVectorFieldSeparated(&magnetization, buf_f);
			if (MPI_rank == 0)
			{
				cout << " Time = " << currentTime;
				if (nComponents == 1)
				{
					cout << " SumX=" << resX << " SumY=" << resY << " SumZ=" << resZ << endl;
					res << scientific << currentTime << "; " << scientific << resX << "; " << scientific << resY << "; " << scientific << resZ << endl;
				}
				if (nComponents == 2)
				{
					cout << " SumX1=" << resX1 << " SumY1=" << resY1 << " SumZ1=" << resZ1 << "	";
					cout << " SumX2=" << resX2 << " SumY2=" << resY2 << " SumZ2=" << resZ2 << endl;
					res << scientific << currentTime << "; " << scientific << resX1 << "; " << scientific << resY1 << "; " << scientific << resZ1 << "; " << scientific << resX2 << "; " << scientific << resY2 << "; " << scientific << resZ2 << endl;
				}
			}

		
		}
		/*sprintf(buf, "field_%d.csv", i);
		string fileName = buf;
		WriteVectorField(1, &magnetization, buf);*/

		MPI_Barrier(MPI_COMM_WORLD);
		/*if (i == 0)
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			WriteVectorField(1, &magnetization, "field_0.dat");
			cout << "Done!" << endl;
		}
		if (i == 1600)
		{
			gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
			WriteVectorField(1, &magnetization, "field_1600.dat");
			cout << "Done!" << endl;
		}*/
		
		//gpuErrchk(cudaMemcpy(addMagx_d, magnx_d, num_el*sizeof(double), cudaMemcpyDeviceToDevice));
		//gpuErrchk(cudaMemcpy(addMagy_d, magny_d, num_el*sizeof(double), cudaMemcpyDeviceToDevice));
		//gpuErrchk(cudaMemcpy(addMagz_d, magnz_d, num_el*sizeof(double), cudaMemcpyDeviceToDevice));
		//NMR_2d_basic << < dimGrid2d, dimBlock >> > (actCell_d, diff_d,
		//	magnx_d, magny_d, magnz_d,
		//	mFx_d, mFy_d, mFz_d,
		//	nFx_d, nFy_d, nFz_d,
		//	velx_d, vely_d, velz_d,
		//	addMagx_d, addMagy_d, addMagz_d,
		//	dTime, dx, dy, dz, gs,
		//	nVectorMult, nVelocity, nDiffusion, relT1_d, relT2_d, equilibriumMagnetization);
		//gpuErrchk(cudaMemcpy(magnx_d, addMagx_d, num_el*sizeof(double), cudaMemcpyDeviceToDevice));
		//gpuErrchk(cudaMemcpy(magny_d, addMagy_d, num_el*sizeof(double), cudaMemcpyDeviceToDevice));
		//gpuErrchk(cudaMemcpy(magnz_d, addMagz_d, num_el*sizeof(double), cudaMemcpyDeviceToDevice));
		////_Maccormak
		//if (nVelocity == 1)
		//{
		//	NMR_2d_predictor << < dimGrid2d, dimBlock >> > (actCell_d,
		//		magnx_d, magny_d, magnz_d,
		//		predictorX, predictorY, predictorZ,
		//		velx_d, vely_d, velz_d,
		//		dTime, dx, dy, dz, gs);
		//	BoundaryConditionsForVectorField_gpu(predictorX, predictorY, predictorZ, &magnetization, gs);
		//	NMR_2d_corrector << < dimGrid2d, dimBlock >> >(actCell_d,
		//		magnx_d, magny_d, magnz_d,
		//		predictorX, predictorY, predictorZ,
		//		correctorX, correctorY, correctorZ,
		//		velx_d, vely_d, velz_d,
		//		dTime, dx, dy, dz, gs);
		//	BoundaryConditionsForVectorField_gpu(correctorX, correctorY, correctorZ, &magnetization, gs);
		//	gpuErrchk(cudaMemcpy(magnx_d, correctorX, num_el*sizeof(double), cudaMemcpyDeviceToDevice));
		//	gpuErrchk(cudaMemcpy(magny_d, correctorY, num_el*sizeof(double), cudaMemcpyDeviceToDevice));
		//	gpuErrchk(cudaMemcpy(magnz_d, correctorZ, num_el*sizeof(double), cudaMemcpyDeviceToDevice));
		//	NMR_2d_softner << <dimGrid2dM1, dimBlock >> >(actCell_d,
		//		magnx_d, magny_d, magnz_d,
		//		correctorX, correctorY, correctorZ,
		//		velx_d, vely_d, velz_d,
		//		dTime, dx, dy, dz, gs);

		//}

		
		
	}
	if (MPI_rank == 0) res.close();
	sprintf(buf, "results_NMR/results_grad_%d_%d_%d.csv", PulseNum[0], PulseNum[1], PulseNum[2]);
	string fileName = buf;
	ofstream resGrad(fileName.c_str());
	gpuErrchk(cudaMemcpy(magnetization.x, magnx_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(magnetization.y, magny_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(magnetization.z, magnz_d, (gs.ik + 2)*(gs.jk + 2)*(gs.kk + 2)*sizeof(double), cudaMemcpyDeviceToHost));
	OutputX(&magnetization);
	OutputY(&magnetization);
	OutputZ(&magnetization);
	lastNum = lastNum + ndPrint;
	sprintf(buf_f, "fields/field_%d_%d_%d_%d", PulseNum[0], PulseNum[1], PulseNum[2], lastNum);
	//WriteVectorFieldSeparated(&magnetization, buf_f);
	if (MPI_rank == 0)
	{
		if (nComponents == 1)
		{
			cout << " SumX=" << resX << " SumY=" << resY << " SumZ=" << resZ << endl;
			resGrad << scientific << currentTime << "; " << scientific << resX << "; " << scientific << resY << "; " << scientific << resZ << endl;
		}
		if (nComponents == 2)
		{
			cout << " SumX1=" << resX1 << " SumY1=" << resY1 << " SumZ1=" << resZ1 << endl;
			cout << " SumX2=" << resX2 << " SumY2=" << resY2 << " SumZ2=" << resZ2 << endl;
			resGrad << scientific << currentTime << "; " << scientific << resX1 << "; " << scientific << resY1 << "; " << scientific << resZ1 << "; " << scientific << resX2 << "; " << scientific << resY2 << "; " << scientific << resZ2 << endl;
		}
	}
	cudaDeviceReset();
	if (MPI_rank == 0) resGrad.close();
	ClearAll();
}

void NmrRelaxationSolve::BoundaryConditionsForVectorField_gpu(double* magnx_d, double* magny_d, double* magnz_d, VectorField* magn, GridSize gs)
{
	dim3 dimBlockBor = dim3(BOR_BLOCK_SIZE, BOR_BLOCK_SIZE);
	dim3 dimGridBor_x = dim3((int)ceil((double)(gs.jk+2) / BOR_BLOCK_SIZE), (int)ceil((double)(gs.kk+2) / BOR_BLOCK_SIZE));
	dim3 dimGridBor_y = dim3((int)ceil((double)(gs.ik+2) / BOR_BLOCK_SIZE), (int)ceil((double)(gs.kk+2) / BOR_BLOCK_SIZE));
	dim3 dimGridBor_z = dim3((int)ceil((double)(gs.ik+2) / BOR_BLOCK_SIZE), (int)ceil((double)(gs.jk+2) / BOR_BLOCK_SIZE));

	if (nBoundary1 == 1 && nBoundary3 == 1)
	{
		BC_ZeroDer_X << <dimGridBor_x, dimBlockBor >> > (magnx_d, gs);
		BC_ZeroDer_X << <dimGridBor_x, dimBlockBor >> > (magny_d, gs);
		BC_ZeroDer_X << <dimGridBor_x, dimBlockBor >> > (magnz_d, gs);

		//BC_Advanced_XM << <dimGridBor_x, dimBlockBor >> > (magny_d, 0.0, gs);
		//BC_Advanced_XP << <dimGridBor_x, dimBlockBor >> > (magny_d, 0.0, gs);
		//BC_Advanced_XP << <dimGridBor_x, dimBlockBor >> > (magny_d, 0.0, gs);
		//BC_Advanced_XM << <dimGridBor_x, dimBlockBor >> > (magny_d, 0.0, gs);

	}

	if (nBoundary2 == 1 && nBoundary4 == 1)
	{
		BC_ZeroDer_Y << <dimGridBor_y, dimBlockBor >> > (magnx_d, gs);
		BC_ZeroDer_Y << <dimGridBor_y, dimBlockBor >> > (magny_d, gs);
		BC_ZeroDer_Y << <dimGridBor_y, dimBlockBor >> > (magnz_d, gs);
	}

	if (nBoundary5 == 1 && nBoundary6 == 1)
	{
		BC_ZeroDer_Z << <dimGridBor_z, dimBlockBor >> > (magnx_d, gs);
		BC_ZeroDer_Z << <dimGridBor_z, dimBlockBor >> > (magny_d, gs);
		BC_ZeroDer_Z << <dimGridBor_z, dimBlockBor >> > (magnz_d, gs);
	}

	if (nBoundary1 == 3 && nBoundary3 == 3)
	{
		BC_Inf_X << <dimGridBor_x, dimBlockBor >> > (magnx_d, gs);
		BC_Inf_X << <dimGridBor_x, dimBlockBor >> > (magny_d, gs);
		BC_Inf_X << <dimGridBor_x, dimBlockBor >> > (magnz_d, gs);
	}

	if (nBoundary2 == 3 && nBoundary4 == 3)
	{
		BC_Inf_Y << <dimGridBor_y, dimBlockBor >> > (magnx_d, gs);
		BC_Inf_Y << <dimGridBor_y, dimBlockBor >> > (magny_d, gs);
		BC_Inf_Y << <dimGridBor_y, dimBlockBor >> > (magnz_d, gs);
	}

	if (nBoundary5 == 3 && nBoundary6 == 3)
	{
		BC_Inf_Z << <dimGridBor_z, dimBlockBor >> > (magnx_d, gs);
		BC_Inf_Z << <dimGridBor_z, dimBlockBor >> > (magny_d, gs);
		BC_Inf_Z << <dimGridBor_z, dimBlockBor >> > (magnz_d, gs);
	}

	if (nBoundary1 == 2)
	{
		BC_Advanced_XM << <dimGridBor_x, dimBlockBor >> > (magnx_d, 0.0, gs);
		BC_Advanced_XM << <dimGridBor_x, dimBlockBor >> > (magny_d, 0.0, gs);
		BC_Advanced_XM << <dimGridBor_x, dimBlockBor >> > (magnz_d, 0.0, gs);

		BC_Advanced_XP << <dimGridBor_x, dimBlockBor >> > (magnx_d, 0.0, gs);
		BC_Advanced_XP << <dimGridBor_x, dimBlockBor >> > (magny_d, 0.0, gs);
		BC_Advanced_XP << <dimGridBor_x, dimBlockBor >> > (magnz_d, 0.0, gs);
	}

	if (nBoundary3 == 2)
	{
		BC_Advanced_XP << <dimGridBor_x, dimBlockBor >> > (magnx_d, 0.0, gs);
		BC_Advanced_XP << <dimGridBor_x, dimBlockBor >> > (magny_d, 0.0, gs);
		BC_Advanced_XP << <dimGridBor_x, dimBlockBor >> > (magnz_d, 0.0, gs);

		BC_Advanced_XM << <dimGridBor_x, dimBlockBor >> > (magnx_d, 0.0, gs);
		BC_Advanced_XM << <dimGridBor_x, dimBlockBor >> > (magny_d, 0.0, gs);
		BC_Advanced_XM << <dimGridBor_x, dimBlockBor >> > (magnz_d, 0.0, gs);
	}

	/*
	Check logic for  exchange!!!
	//gpuErrchk(cudaMemcpy(magn->x + (gs.ik + 2)*(gs.jk + 2), magnx_d + (gs.ik + 2)*(gs.jk + 2), (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(magn->y + (gs.ik + 2)*(gs.jk + 2), magny_d + (gs.ik + 2)*(gs.jk + 2), (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(magn->z + (gs.ik + 2)*(gs.jk + 2), magnz_d + (gs.ik + 2)*(gs.jk + 2), (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(magn->x + (gs.kk)*(gs.ik + 2)*(gs.jk + 2), magnx_d + (gs.kk)*(gs.ik + 2)*(gs.jk + 2), (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(magn->y + (gs.kk)*(gs.ik + 2)*(gs.jk + 2), magny_d + (gs.kk)*(gs.ik + 2)*(gs.jk + 2), (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(magn->z + (gs.kk)*(gs.ik + 2)*(gs.jk + 2), magnz_d + (gs.kk)*(gs.ik + 2)*(gs.jk + 2), (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyDeviceToHost));

	////MPI boundary conditions
	//MPI_Status status;
	//for (int j = 0; j < (gs.jk + 2); j++)
	//{
	//	for (int i = 0; i < (gs.ik + 2); i++)
	//	{
	//		MPI_Sendrecv(&magn->x[index(i, j, gs.kk, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
	//			&magn->x[index(i, j, 0, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
	//			MPI_COMM_WORLD, &status);
	//		MPI_Sendrecv(&magn->x[index(i, j, 1, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
	//			&magn->x[index(i, j, gs.kk + 1, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
	//			MPI_COMM_WORLD, &status);
	//	}
	//}
	//for (int j = 0; j < (gs.jk + 2); j++)
	//{
	//	for (int i = 0; i < (gs.ik + 2); i++)
	//	{
	//		MPI_Sendrecv(&magn->y[index(i, j, gs.kk, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
	//			&magn->y[index(i, j, 0, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
	//			MPI_COMM_WORLD, &status);
	//		MPI_Sendrecv(&magn->y[index(i, j, 1, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
	//			&magn->y[index(i, j, gs.kk + 1, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
	//			MPI_COMM_WORLD, &status);
	//	}
	//}
	//for (int j = 0; j < (gs.jk + 2); j++)
	//{
	//	for (int i = 0; i < (gs.ik + 2); i++)
	//	{
	//		MPI_Sendrecv(&magn->z[index(i, j, gs.kk, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
	//			&magn->z[index(i, j, 0, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
	//			MPI_COMM_WORLD, &status);
	//		MPI_Sendrecv(&magn->z[index(i, j, 1, gs)], 1, MPI_DOUBLE, (MPI_rank + MPI_size - 1) % MPI_size, MPI_rank,
	//			&magn->z[index(i, j, gs.kk + 1, gs)], 1, MPI_DOUBLE, (MPI_rank + 1) % MPI_size, (MPI_rank + 1) % MPI_size,
	//			MPI_COMM_WORLD, &status);
	//	}
	//}
	//end of MPI boundary conditions. Corrections within 0 & size-1 ranks

	//gpuErrchk(cudaMemcpy(magnx_d, magn->x, (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(magny_d, magn->y, (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(magnz_d, magn->z, (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyHostToDevice));

	//gpuErrchk(cudaMemcpy(magnx_d + (gs.kk + 1)*(gs.ik + 2)*(gs.jk + 2), magn->x + (gs.kk + 1)*(gs.ik + 2)*(gs.jk + 2), (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(magny_d + (gs.kk + 1)*(gs.ik + 2)*(gs.jk + 2), magn->y + (gs.kk + 1)*(gs.ik + 2)*(gs.jk + 2), (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(magnz_d + (gs.kk + 1)*(gs.ik + 2)*(gs.jk + 2), magn->z + (gs.kk + 1)*(gs.ik + 2)*(gs.jk + 2), (gs.ik + 2)*(gs.jk + 2) *sizeof(double), cudaMemcpyHostToDevice));
	TO CHECK!!!!!!!!!!!!!
	

	if (MPI_rank == 0)
	{
		NMR_bc_z_zm_d <<<dimGridBor_z, dimBlockBor >>> (magnx_d, gs);
		NMR_bc_z_zm_d <<<dimGridBor_z, dimBlockBor >>> (magny_d, gs);
		NMR_bc_z_zm_d <<<dimGridBor_z, dimBlockBor >>> (magnz_d, gs);
	}
	if (MPI_rank == (MPI_size - 1))
	{
		NMR_bc_z_zp_d <<<dimGridBor_z, dimBlockBor >>> (magnx_d, gs);
		NMR_bc_z_zp_d <<<dimGridBor_z, dimBlockBor >>> (magny_d, gs);
		NMR_bc_z_zp_d <<<dimGridBor_z, dimBlockBor >>> (magnz_d, gs);
	}
	*/

}

void NmrRelaxationSolve::ExplicitMakUpgrade()
{
	dim3 dimBlock = dim3(CUDA_THREAD_PER_BLOCK);
	dim3 dimGrid2d = dim3((int)(ceil((gs.ik)*(double)(gs.jk) / CUDA_THREAD_PER_BLOCK)), gs.kk);

	NMR_predictor_upgrade << < dimGrid2d, dimBlock >> > (actCell_d, diff_d,
		magnx_d, magny_d, magnz_d,
		mFx_d, mFy_d, mFz_d,
		nFx_d, nFy_d, nFz_d,
		predictorX, predictorY, predictorZ,
		correctorX, correctorY, correctorZ,
		velx_d, vely_d, velz_d,
		dTime, dx, dy, dz, gs,
		nVectorMult, nVelocity, nDiffusion, relT1_d, relT2_d, equilibriumMagnetization);
	BoundaryConditionsForVectorField_gpu(predictorX, predictorY, predictorZ, &magnetization, gs);
	NMR_corrector_upgrade << < dimGrid2d, dimBlock >> >(actCell_d, diff_d,
		magnx_d, magny_d, magnz_d,
		mFx_d, mFy_d, mFz_d,
		nFx_d, nFy_d, nFz_d,
		predictorX, predictorY, predictorZ,
		correctorX, correctorY, correctorZ,
		velx_d, vely_d, velz_d,
		dTime, dx, dy, dz, gs,
		nVectorMult, nVelocity, nDiffusion, relT1_d, relT2_d, equilibriumMagnetization);
	BoundaryConditionsForVectorField_gpu(correctorX, correctorY, correctorZ, &magnetization, gs);
	gpuErrchk(cudaMemcpy(magnx_d, correctorX, gs.num*sizeof(double), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(magny_d, correctorY, gs.num*sizeof(double), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(magnz_d, correctorZ, gs.num*sizeof(double), cudaMemcpyDeviceToDevice));
}

#ifndef gpuasserth
#define gpuasserth

#include <cuda_runtime.h>
#include <string>
#include <sstream>

#define gpuErrchk(ans) { gpuAssert ((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code!=cudaSuccess)
	{
		std::string tmpstr=cudaGetErrorString(code);
		std::ostringstream ss;
		ss << std::dec << line;
		tmpstr="GPUassert:"+tmpstr+file + ss.str();
		ss.clear();
		std::cout<<tmpstr<<std::endl<<std::endl;
		//fprintf(stderr,"GPUassert: %s %s %d\n",cudaGetErrorString(code), file,line);
		if (abort) exit(code);
	}
};

#endif


















