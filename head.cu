__global__ void	setup_kernel(unsigned long long seed,cru2 *CRU2);
__global__ void Initialize( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, 
						double *cnsr, double *cnsrnext, double *cs, double *csnext, double cjsr_b);
__global__ void Compute( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, 
						double *cnsr, double *cnsrnext, double *cs, double *csnext, double v, int step, double Ku, 
						double Kb, double cpstar, double xnai, double sv_ica, double sv_ncx);

__device__ int RyR_gating (curandState *state, double Ku, double Kb, double cpstar, double cp, double cjsr, int * ncu, int * nou, 
							int * ncb, int * nob, int i, int j, int k);
__device__ int LTCC_gating(curandState *state,int i, double cpx,double v);
__device__ double single_LTCC_current(double v, double cp);

__device__ double NCX(double cs, double v,double tperiod, double xnai, double *Ancx);
__device__ double Uptake(double ci, double cnsr);

__device__ double MyoCa(double CaMyo, double MgMyo, double calciu, double dt);
__device__ double MyoMg(double CaMyo, double MgMyo, double calciu, double dt);
__device__ double TroPf(double CaTf, double calciu, double dt);
__device__ double TroPs(double CaTs, double calciu, double dt);
__device__ double buCal(double CaCal, double calciu, double dt);
__device__ double buDye(double CaDye, double calciu, double dt);
__device__ double buSR(double CaSR, double calciu, double dt);
__device__ double buSar(double CaSar, double calciu, double dt);
__device__ double buSarH(double CaSarh, double calciu, double dt);
__device__ double buSarj(double CaSar, double calciu, double dt);
__device__ double buSarHj(double CaSarh, double calciu, double dt);