void Initialize( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, 
						double *cnsr, double *cnsrnext, double *cs, double *csnext, double cjsr_b);
void Compute( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, 
						double *cnsr, double *cnsrnext, double *cs, double *csnext, double v, int step, double Ku, 
						double Kb, double cpstar, double xnai, double sv_ica, double sv_ncx);

int RyR_gating (double Ku, double Kb, double cpstar, double cp, double cjsr, int * ncu, int * nou, 
							int * ncb, int * nob, int i, int j, int k);
int LTCC_gating(int i, double cpx,double v);
double single_LTCC_current(double v, double cp);

double NCX(double cs, double v,double tperiod, double xnai, double *Ancx);
double Uptake(double ci, double cnsr);

double MyoCa(double CaMyo, double MgMyo, double calciu, double dt);
double MyoMg(double CaMyo, double MgMyo, double calciu, double dt);
double TroPf(double CaTf, double calciu, double dt);
double TroPs(double CaTs, double calciu, double dt);
double buCal(double CaCal, double calciu, double dt);
double buDye(double CaDye, double calciu, double dt);
double buSR(double CaSR, double calciu, double dt);
double buSar(double CaSar, double calciu, double dt);
double buSarH(double CaSarh, double calciu, double dt);
double buSarj(double CaSar, double calciu, double dt);
double buSarHj(double CaSarh, double calciu, double dt);