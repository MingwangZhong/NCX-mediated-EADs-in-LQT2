/*---------------------------------------------------------------
*
* This code is based on the original Restrepo model, and is improved
* by CIRCS group of Northeastern University.
*
* Contact Information:
* 
* Center for interdisciplinary research on complex systems
* Departments of Physics, Northeastern University
* 
* Alain Karma		a.karma (at) northeastern.edu
*
* The code is used to reproduce results in
*
* Zhong, Mingwang, Colin M. Rees, Dmitry Terentyev, Bum-Rak Choi, 
* Gideon Koren, and Alain Karma. "NCX-mediated subcellular Ca2+ 
* dynamics underlying early afterdepolarizations in LQT2 cardiomyocytes." 
* Biophysical journal 115, no. 6 (2018): 1019-1032.
*--------------------------------------------------------------- */

/*---------------------------------------------------------------
* The original Restrepo model:
*
* Restrepo, Juan G., James N. Weiss, and Alain Karma. 
* "Calsequestrin-mediated mechanism for cellular calcium transient 
* alternans." Biophysical journal 95, no. 8 (2008): 3767-3789.
*--------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
#define LQT2			 //Long-QT 2 syndrome simulation. No I_Kr
#define ISO			// isoproterenol, increases uptake, I_Ca,L
//#define permeabilized			//Permeabalized cell. No sarcolemmal ion channels

//#define vclamp		//step function voltage clamp
#ifdef vclamp
	#define clampvoltage ( atof(argv[3]) )
#endif

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
#define nbeat 15	//number of pacing beat
#define PCL 4000.0	//ms, pacing cycle length
#define DT 0.025 //ms, time for each step

#define OUT_STEP 100 // number of steps to output data
#define outt (OUT_STEP*DT) //ms, time interval to output data
//#define outputlinescan // output line scan

#define Vp 0.00126 	//um^3, Volume of the proximal space
#define Vs 0.025 //um^3,  volume of submembrane space
#define Vjsr 0.02		//um^3, Volume of the Jsr space
#define Vi (0.5/8.0)		//um^3, Volume of the Local cytosolic, divided into 8 conpartments
#define Vnsr (0.025/8.0)	//um^3, Volume of the Local Nsr space, divided into 8 conpartments
#define taups 0.0283 //ms, diffusion time scale between proximal and submembrane spaces
#define taupi 0.1	//ms, diffusion time scale between proximal space and cytosol
#define tausi 0.04	//ms, diffusion time scale between submembrane space and cytosol
#define taust 1.42 //ms, diffusion time scale for submembrane space along t-tubules
#define tautr 6.25	//ms, diffusion time scale between NSR and JSR 
#define taunl 4.2			//ms, diffusion time scale of NSR in longitudinal direction
#define taunt 1.26			//ms, diffusion time scale of NSR in transverse direction
#define tauil 0.98 //ms, cytosol diffusion time scale in longitudinal direction
#define tauit 0.462 //ms, cytosol diffusion time scale in transverse direction

#ifdef permeabilized
	#define ci_basal 0.3 // initial value of ci
	#define cjsr_basal 900 // initial value of cjsr
#else
	#define ci_basal 0.0944
	#define cjsr_basal ( atof(argv[4]) ) //560.0 for a patch clamp 1 sec wait to get to 645
#endif

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
//////////// ion channel parameters 

// objects starting with sv basically denote the prefactors of ion channel current.
#define sviks 1.0 // prefactor of IKs
#define svtof 1.0 // Itof
#define svtos 1.0 // Itos
#define svnak 1.0 // INaK
#define svk1  1.0 // IK1
#define svileak 1.0 // I_leak, between NSR and cytosol
#define svtauxs 1.2 // for IKs activation time scale

#ifdef LQT2
	#define svikr 0
#else 
	#define svikr 1.0
#endif

#ifdef 	ISO
	#define sviup 1.75
#else
	#define svipu  1.0
#endif

// ryr gating
#define nryr (100)		//Number of RyR channels
#define svjmax 11.5     // J_max prefactor
#define tauu (2000.0) //ms, Unbinding transition time
#define taub (2.0) //ms, Binding transition time 
#define taucb 1.0 //ms, transition time from open bound state to closed bound state
#define taucu 1.0 //ms, transition time from open unbound state to closed unbound state

// LCC ica
#define	svica (atof(argv[8])) // prefactor of ICa
#define icagamma (0.0) // if LTCC locates at submembrane space. 0 means no.
#define Pca 17.85   // umol/C/ms, strength of single channel current
#define gammai 0.341
#define gammao 0.341
#define svncp 4 // total number of LTCC channels in each CRU

// luminal gating
#define nCa 22.0 // number of Ca2+ binding sites of CSQN molecular
#define ratedimer 5000.0
#define kdimer 850.0
#define hilldimer 23.0
#define BCSQN 460.0 //uM, total concentration of CSQN
#define kbers 600.0


// relate to Na+ or K+: Incx, Inak
#define svncx (atof(argv[9])) // prefactor of NCX current
#define xnao 140.0 //mM, external sodium concentration, [Na+]o
#define vnaca 21.0 // strength of NCX
#define Kmcai 0.00359
#define Kmcao 1.3
#define Kmnai 12.3
#define Kmnao 87.5
#define eta 0.35
#define ksat 0.27


// other
#define Cext (1*1.8)    // mM, external Ca2+ concentration
#define Cm 45	// capacitance of the cell membrane

#define Farad 96.485    //  C/mmol, Faraday constant
#define xR  8.314   //  J/mol/K
#define Temper  308      //K, temperature
#define frt (96.485/8.314/308.0)
#define pi 3.1415926


//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
////////////////// CUDA block size 
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 7
#define BLOCK_SIZE_Z 4

#define nx 64				//Number of CRUs in the x direction
#define ny 28				//Number of CRUs in the y direction
#define nz 12				//Number of CRUs in the z direction

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
#define pos(x,y,z)	((nx*ny)*(z)+nx*(y)+(x)) // position of a CRU
#define pow6(x) ((x)*(x)*(x)*(x)*(x)*(x))
#define pow4(x) ((x)*(x)*(x)*(x))
#define pow3(x) ((x)*(x)*(x))
#define pow2(x) ((x)*(x))


typedef struct{
	double casar; // Ca2+ bound sarcolemma buffer in submembrane
	double casarh; // Membrane/High in submembrane
	double casarj; // Ca2+ bound sarcolemma buffer in dyad
	double casarhj;// Membrane/High in dyad
}sl_bu;

typedef struct{ // sytosolic buffers
	double cacal; // Ca bound Calmodulin
	double catf; // Ca bound fast Troponin
	double cats; // Ca bound slow Troponin
	double casr; // Ca bound SR buffer
	double camyo; // Ca bound Myosin
	double mgmyo;// Mg bound Myosin
	double cadye; // Ca bound dye
}cyt_bu;

typedef struct{
	double xiup; //uM/ms, uptake current
	double xileak; //uM/ms, leak current
}cytosol;

typedef struct{
	int nsign;
	int nspark;
	double randomi;
	double cp; //uM
	double cpnext; //uM
	double cjsr; //uM
	double Tcj; //uM total Ca2+ in JSR
	
	double xire; // flux of RyR release
	int lcc[8]; // state of each LTCC
	int nl;
	int nou; // number of RyR channels in open unbound state
	int ncu; // closed unbound
	int nob; // open bound 
	int ncb; // closed bound 
	double po; // (nou+nob)/nryr, fraction of channels in open state 
	int sparknum;
	double Ancx; // allosteric Ca2+ activation of NCX
}cru2;
	
typedef struct{
	double xinaca; //uM/ms, flux of NCX
	double xica; //uM/ms, flux of ICa
}cru;

#include "head.c"
#include "subroutine.c"
#include "routine.c"

int main(int argc, char **argv)
{
	// Allocate arrays memory
	size_t ArraySize_cru = nx*ny*nz*sizeof(cru);
	size_t ArraySize_cru2 = nx*ny*nz*sizeof(cru2);
	size_t ArraySize_cyt = 8*nx*ny*nz*sizeof(cytosol);
	size_t ArraySize_cbu = 8*nx*ny*nz*sizeof(cyt_bu);
	size_t ArraySize_sbu = nx*ny*nz*sizeof(sl_bu);
	size_t ArraySize_dos = nx*ny*nz*sizeof(double);
	size_t ArraySize_dol = 8*nx*ny*nz*sizeof(double);

	cru *CRU = (cru*)malloc(ArraySize_cru);
	cru2 *CRU2 = (cru2*)malloc(ArraySize_cru2);
	cytosol *CYT = (cytosol*)malloc(ArraySize_cyt);
	cyt_bu *CBU = (cyt_bu*)malloc(ArraySize_cbu);
	sl_bu *SBU = (sl_bu*)malloc(ArraySize_sbu);
	double *ci = (double*)malloc(ArraySize_dol);
	double *cinext = (double*)malloc(ArraySize_dol);
	double *cnsr = (double*)malloc(ArraySize_dol);
	double *cnsrnext = (double*)malloc(ArraySize_dol);
	double *cs = (double*)malloc(ArraySize_dos);
	double *csnext = (double*)malloc(ArraySize_dos);

	// output files
	FILE * wholecell_file = fopen("wholecell2.txt","w");
	FILE * output_csNCX20 = fopen("csNCX20ms.txt","w");
	FILE * output_csNCX322 = fopen("csNCX322ms.txt","w");
	FILE * output_csNCX350 = fopen("csNCX350ms.txt","w");
	
	#ifdef outputlinescan
		FILE * linescan_file = fopen("linescan.txt","w");
	#endif

	/////////////////////////////////// variables /////////////////////////////////////////////////
	int step = 0;	// running steps
	double t=0.0;			//time 
	int jx;					 //Loop variable
	int jy;					 //Loop variable
	int jz;					 //Loop variable

	double	cproxit;			// average concentration of the proximal space
	double	csubt;			// average concentration of the submembrane space
	double	cit;				// average concentration of the cytosolic space
	double	cjsrt;			// average concentration of the JSR space
	double	cjt;	// total Ca2+ in the jsR
	double	TotalCa=0; // whole cell total Ca2+, used to check Ca2+ conservation
	double	TotalCa_before = 181.7; // whole cell total Ca2+, used to check Ca2+ conservation
	double	CaExt =0; // Ca2+ influx through ICa and INCX, used to check Ca2+ conservation
	double	cnsrt;			// average concentration of the NSR space
	double	xicatto;			// whole cell Lcc calcium current in the proximal space
	double	out_ica;			 // LCC Strength averaged over output period
	double	out_ina;			 // I_na Strength averaged over output period
	double	xinacato;			// whole cell NCX current in the submembrane space
	double	poto;				// average open probability of RyR channels

	double Ku = atof(argv[5]);  // RyR gating parameter, corresponding to \bar(k)_{p,U} in the paper
	double Kb = atof(argv[6]);  // RyR gating parameter corresponding to \bar(k)_{p,B} in the paper
	double cpstar = atof(argv[7]);  // RyR gating parameter: cp*

	double xki=140.0;	//mM, internal K
	double xko=5.40;	//mM, external K
	double xnai=atof(argv[10]); // intracelluar sodium concentration: [Na+]i
	double v=-80.00;	// voltage
	double xm=0.0010;	// sodium m-gate
	double xh=1.00;	// sodium h-gate
	double xj=1.00;	// soium	j-gate=
	double xr=0.00;	// ikr gate variable
	double xs1=0.08433669901; // iks gate variable
	double xs2=xs1;//0.1412866149; //removed, and replaced with xs1..not sure why
	double qks=0.20;	// iks gate variable
	double xtos=0.010; // ito slow activation
	double ytos=1.00;	// ito slow inactivation
	double xtof=0.020; // ito fast activation
	double ytof=0.80;	// ito slow inactivation
	double xinak; // Inak current
	double xina; // Ina current

	int sparksum = 0; // spark rate
	int Nxyz = (nx-2)*(ny-2)*(nz-2);

	double start_time=clock()/(1.0*CLOCKS_PER_SEC),    end_time; // to calculate running time of the simulation


	////////////////////////////////////////////////////////////////////////////////////////////////
	Initialize(CRU, CRU2, CYT, CBU, SBU, ci, cinext, cnsr, cnsrnext, cs, csnext, cjsr_basal);

	while ( t < nbeat*PCL+100 )
	{
		Compute( CRU, CRU2, CYT, CBU, SBU, ci, cinext, cnsr, cnsrnext, 
					cs, csnext, v, step, Ku, Kb, cpstar, xnai, svica, svncx );
		// update variables
		double *tempci, *tempcs, *tempcnsr;
		tempci = cinext; 		cinext = ci;		ci=tempci;
		tempcs = csnext; 		csnext = cs;		cs=tempcs;
		tempcnsr = cnsrnext;	cnsrnext = cnsr;	cnsr=tempcnsr;

		////////////////////////////
		
		csubt=0;
		xicatto=0;
		xinacato=0;

		for (jz = 1; jz < nz-1; jz++)
			for (jy = 1; jy < ny-1; jy++)
				for (jx = 1; jx < nx-1; jx++) 
				{
					csubt += cs[pos(jx,jy,jz)];
					xicatto=xicatto+CRU[pos(jx,jy,jz)].xica;
					xinacato=xinacato+CRU[pos(jx,jy,jz)].xinaca;

					CaExt = CaExt - CRU[pos(jx,jy,jz)].xica*Vp*DT*(1+icagamma)/(Nxyz) // Ca2+ through ICa and NCX
						+ CRU[pos(jx,jy,jz)].xinaca*Vs*DT/(Nxyz);
				}
		
		csubt=csubt/(Nxyz);
		xicatto=xicatto/Cm*0.0965*Vp*2.0*(icagamma+1.0);
		xinacato=xinacato/Cm*0.0965*Vs;

		out_ica += xicatto;

		//////////////////////////////////// Sodium current: Ina /////////////////////////////////////////
				
		double ena = (1.0/frt)*log(xnao/xnai);		// sodium reversal potential
		double am = 0.32*(v+47.13)/(1.0-exp(-0.1*(v+47.13)));
		double bm = 0.08*exp(-v/11.0);
				
		double ah,bh,aj,bj;
				
		if(v < -40.0)
		{
			ah = 0.135*exp((80.0+v)/(-6.8));
			bh = 3.56*exp(0.079*v)+310000.0*exp(0.35*v);
			aj = (-127140.0*exp(0.2444*v)-0.00003474*exp(-0.04391*v))*((v+37.78)/(1.0+exp(0.311*(v+79.23))));
			bj = (0.1212*exp(-0.01052*v))/(1.0+exp(-0.1378*(v+40.14)));
			//aj=ah; //make j just as h
			//bj=bh; //make j just as h
		}
		else
		{
			ah = 0.00;
			bh = 1.00/(0.130*(1.00+exp((v+10.66)/(-11.10))));
			aj = 0.00;
			bj = (0.3*exp(-0.00000025350*v))/(1.0 + exp(-0.10*(v+32.00)));
			//aj=ah; //make j just as h
			//bj=bh; //make j just as h
		}
				
		double tauh = 1.00/(ah+bh);
		double tauj = 1.00/(aj+bj);
		double taum = 1.00/(am+bm);
				
		double gna = 12.00;			// sodium conductance (mS/micro F)
		double gnaleak = 0.3e-3*5;	//sodium leak conductance
		double gnal = 0.012;	//late sodium conductance (mS/micro F)
		double f_NaL = atof(argv[11]);
		xina = gna*(f_NaL+(1.0-f_NaL)*xh)*(f_NaL+(1.0-f_NaL)*xj)*xm*xm*xm*(v-ena) + gnaleak*(v-ena) ;
				
		xh = ah/(ah+bh)-((ah/(ah+bh))-xh)*exp(-DT/tauh);
		xj = aj/(aj+bj)-((aj/(aj+bj))-xj)*exp(-DT/tauj);
		xm = am/(am+bm)-((am/(am+bm))-xm)*exp(-DT/taum);
		
		out_ina += xina ;

		////////////////////////////// Ikr following Shannon ////////////////////////////////////

		double ek = (1.00/frt)*log(xko/xki);				// K reversal potential = -86.26
		double gss = sqrt(xko/5.40);
		double xkrv1 = 0.001380*(v+7.00)/( 1.0-exp(-0.123*(v+7.00))	);
		double xkrv2 = 0.000610*(v+10.00)/(exp( 0.1450*(v+10.00))-1.00);
		double taukr = 1.00/(xkrv1+xkrv2);
		double xkrinf = 1.00/(1.00+exp(-(v+50.00)/7.50));
		double rg = 1.00/(1.00+exp((v+33.00)/22.40));
				
		double gkr = 0.0078360;	// Ikr conductance
		double xikr = svikr*gkr*gss*xr*rg*(v-ek);
				
		xr = xkrinf-(xkrinf-xr)*exp(-DT/taukr);
		
		////////////////////////////// Iks modified from Shannon, with new Ca dependence /////////////

		double prnak = 0.0183300;
		double qks_inf = 0.2*(1+0.8/(1+pow((0.28/csubt),3)));//0.60*(1.0*csubt);
		double tauqks=1000.00;
				
		double eks = (1.00/frt)*log((xko+prnak*xnao)/(xki+prnak*xnai));
		double xs1ss = 1.0/(1.0+exp(-(v-1.500)/16.700));
				
		double tauxs = svtauxs/(0.0000719*(v+30.00)/(1.00-exp(-0.1480*(v+30.0)))+0.0001310*(v+30.00)/(exp(0.06870*(v+30.00))-1.00));
		double gksx=0.2000; // Iks conductance
		double xiks = sviks*gksx*qks*xs1*xs2*(v-eks);
				
		xs1=xs1ss-(xs1ss-xs1)*exp(-DT/tauxs);
		xs2=xs1ss-(xs1ss-xs2)*exp(-DT/tauxs);
		qks=qks+DT*(qks_inf-qks)/tauqks;
				
		///////////////////////////////	Ik1 following Luo-Rudy formulation (from Shannon model)

		double gkix = 0.600; // Ik1 conductance
		double gki = gkix*(sqrt(xko/5.4));
		double aki = 1.02/(1.0+exp(0.2385*(v-ek-59.215)));
		double bki = (0.49124*exp(0.08032*(v-ek+5.476))+exp(0.061750*(v-ek-594.31)))/(1.0+exp(-0.5143*(v-ek+4.753)));
		double xkin = aki/(aki+bki);
		double xik1 = svk1*gki*xkin*(v-ek);	 
		
		/////////////////////////////// Ito slow following Shannon et. al. 2005 ///////////////////////

		double rt1 = -(v+3.0)/15.00;
		double rt2 = (v+33.5)/10.00;
		double rt3 = (v+60.00)/10.00;
		double xtos_inf = 1.00/(1.0+exp(rt1));
		double ytos_inf = 1.00/(1.00+exp(rt2));
		double rs_inf = 1.00/(1.00+exp(rt2));
		double txs = 9.00/(1.00+exp(-rt1)) + 0.50;
		double tys = 3000.00/(1.0+exp(rt3)) + 30.00; //cmrchange
		double gtos=0.040; // ito slow conductance
				
		double xitos = svtos*gtos*xtos*(ytos+0.50*rs_inf)*(v-ek); // ito slow
				
		xtos = xtos_inf-(xtos_inf-xtos)*exp(-DT/txs);
		ytos = ytos_inf-(ytos_inf-ytos)*exp(-DT/tys);
		
		//////////////////////////////// Ito fast following Shannon et. al. 2005 /////////////////////////

		double xtof_inf = xtos_inf;
		double ytof_inf = ytos_inf;
		
		double rt4 = -(v/30.00)*(v/30.00);
		double rt5 = (v+33.50)/10.00;
		double txf = 3.50*exp(rt4)+1.50;
		double tyf = 20.0/(1.0+exp(rt5))+20.00;
		double gtof = 0.10;	//! ito fast conductance
				
		double xitof = svtof*gtof*xtof*ytof*(v-ek);
				
		xtof = xtof_inf-(xtof_inf-xtof)*exp(-DT/txf);
		ytof = ytof_inf-(ytof_inf-ytof)*exp(-DT/tyf);

		
		////////////////////////////////	Inak (sodium-potassium exchanger) following Shannon ////////////
				
		double xkmko = 1.50;
		double xkmnai = 12.00;
		double xibarnak = 1.5000;
		double hh = 1.00;	// Na dependence exponent

		double sigma = (exp(xnao/67.30)-1.00)/7.00;
		double fnak = 1.00/(1.0+0.1245*exp(-0.1*v*frt)+0.0365*sigma*exp(-v*frt));
		xinak = svnak * xibarnak*fnak*(1.0/(1.0+pow((xkmnai/xnai),hh)))*xko/(xko+xkmko);


		//////////////////////////////////// sodium dynamics ////////////////////////////////////////
		double wcainv = 1.0/50.0;		//! conversion factor between pA to micro //! amps/ micro farads
		double conv = 0.18*12500;	//conversion from muM/ms to pA	(includes factor of 2 for Ca2+)
		double xinaca = xinacato;	//convert ion flow to current:	net ion flow = 1/2	calcium flow
				
		double trick=1.0;
		double xrr=trick*(1.0/wcainv/conv)/1000.0; // note: sodium is in m molar, so need to divide by 1000
		double dnai = -xrr*(xina +3.0*xinak+3.0*xinaca);
		xnai += dnai*DT;

		/////////////////////////////////////////////////////////////////////////////////////////////
		double stim;
		if( fmod(t+PCL-100,PCL) < 1.0 ) // duration of stimulation: 1 ms
			stim = 80.0;
		else						
			stim= 0.0;

		#ifdef permeabilized
				stim = 0.0;
		#endif
		
		////////////////////////////////////////// dv/dt ////////////////////////////////////////////
				
		double dvh = -( xina + xik1 + xikr + xiks + xitos + xitof + xinacato + xicatto + xinak ) + stim; 
		v += dvh*DT;


		#ifdef vclamp
			v=-86;
			if( t > 100 )	//allow 1 second of rest before voltage clamp is applied
				v = clampvoltage;
		#endif


		////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////// output
		if ( step%OUT_STEP==0 )
		{	
			cit = 0;
			cproxit=0;
			cjsrt=0;
			cjt=0;
			cnsrt=0;
			poto=0;

			double pbcto=0;
			double csub2t=0;
			double csub3t=0;
			double ncxfwd=0;

			double isit=0;
			double catft=0;
			double catst=0;
			double casrt=0;
			double camyot=0;
			double mgmyot=0;
			double cacalt=0;
			double cadyet=0;

			double casart = 0;
			double casarht = 0;
			double casarjt = 0;
			double casarhjt = 0;

			double leakt=0;
			double upt=0;
			double ire=0;
			
			int tn1 = 0; // record the state of LTCC
			int tn2 = 0;
			int tn3 = 0;
			int tn4 = 0;
			int tn5 = 0;
			int tn6 = 0;
			int tn7 = 0;
			
			int tnou = 0; // average number of RyR channels in open unbound state
			int tnob = 0;
			int tncu = 0;
			int tncb = 0;

			int tcruo = 0;
			int tcruo2 = 0;
			int tcruo3 = 0;
			int tcruo4 = 0;
			double icaflux = 0;
			double ncxflux = 0;

			double outAncx = 0;
			for (jz = 1; jz < nz-1; jz++)
			{
				for (jy = 1; jy < ny-1; jy++)
				{
					for (jx = 1; jx < nx-1; jx++) 
					{	
						icaflux=icaflux+CRU[pos(jx,jy,jz)].xica;
						ncxflux=ncxflux+CRU[pos(jx,jy,jz)].xinaca;
						if (	CRU[pos(jx,jy,jz)].xinaca < 0 )
						ncxfwd=ncxfwd+CRU[pos(jx,jy,jz)].xinaca;
						outAncx+=CRU2[pos(jx,jy,jz)].Ancx;


						cproxit=cproxit+CRU2[pos(jx,jy,jz)].cp;
						csub2t=csub2t+cs[pos(jx,jy,jz)]*cs[pos(jx,jy,jz)];
						csub3t=csub3t+cs[pos(jx,jy,jz)]*cs[pos(jx,jy,jz)]*cs[pos(jx,jy,jz)];
						cjsrt=cjsrt+CRU2[pos(jx,jy,jz)].cjsr;
						cjt=cjt+CRU2[pos(jx,jy,jz)].Tcj;
						poto=poto+CRU2[pos(jx,jy,jz)].po;
						pbcto=pbcto+CRU2[pos(jx,jy,jz)].ncb;

						casart += SBU[pos(jx,jy,jz)].casar;
						casarht += SBU[pos(jx,jy,jz)].casarh;
						casarjt += SBU[pos(jx,jy,jz)].casarj;
						casarhjt += SBU[pos(jx,jy,jz)].casarhj;

						for ( int ii = 0; ii < 8; ++ii )
						{
							cit+=ci[pos(jx,jy,jz)*8+ii]/8.;
							cnsrt=cnsrt+cnsr[pos(jx,jy,jz)*8+ii]/8.;
							catft= catft+CBU[pos(jx,jy,jz)*8+ii].catf/8.;
							catst= catst+CBU[pos(jx,jy,jz)*8+ii].cats/8.;
							casrt= casrt+CBU[pos(jx,jy,jz)*8+ii].casr/8.;
							camyot= camyot+CBU[pos(jx,jy,jz)*8+ii].camyo/8.;
							mgmyot= mgmyot+CBU[pos(jx,jy,jz)*8+ii].mgmyo/8.;
							cacalt= cacalt+CBU[pos(jx,jy,jz)*8+ii].cacal/8.;
							cadyet= cadyet+CBU[pos(jx,jy,jz)*8+ii].cadye/8.;
							leakt= leakt+CYT[pos(jx,jy,jz)*8+ii].xileak/8.;
							upt=	upt+CYT[pos(jx,jy,jz)*8+ii].xiup/8.;

							if( ci[pos(jx,jy,jz)*8+ii] > 1000 )
							{
								printf("Error!\tci=%g\tt=%g\tjx=%d\tjy=%d\tjz=%d\tii=%d\n",
										ci[pos(jx,jy,jz)*8+ii], t, jx, jy, jz, ii);
							}
						}


						ire += CRU2[pos(jx,jy,jz)].xire;

						tnou += CRU2[pos(jx,jy,jz)].nou;
						tnob += CRU2[pos(jx,jy,jz)].nob;
						tncu += CRU2[pos(jx,jy,jz)].ncu;
						tncb += CRU2[pos(jx,jy,jz)].ncb;

						sparksum += CRU2[pos(jx,jy,jz)].nspark;

						if ( CRU2[pos(jx,jy,jz)].nou + CRU2[pos(jx,jy,jz)].nob > 30 )
							++tcruo;
						if ( CRU2[pos(jx,jy,jz)].nou + CRU2[pos(jx,jy,jz)].nob > 40 )
							++tcruo2;
						if ( CRU2[pos(jx,jy,jz)].nou + CRU2[pos(jx,jy,jz)].nob > 30 || 
							CRU2[pos(jx,jy,jz)].nob + CRU2[pos(jx,jy,jz)].ncb > 50 )
							++tcruo3;
						if ( CRU2[pos(jx,jy,jz)].ncu < 20 )
							++tcruo4;

						for( int jj = 0; jj < 8; ++jj )
						{
							switch (CRU2[pos(jx,jy,jz)].lcc[jj])
							{
								case 1: ++tn1; break;
								case 2: ++tn2; break;
								case 3: ++tn1; ++tn2; break;
								case 4: ++tn3; ++tn5; break;
								case 5: ++tn1; ++tn3; ++tn5; break;
								case 6: ++tn2; ++tn3; ++tn5; break;
								case 7: ++tn1; ++tn2; ++tn3; ++tn5; break;
								case 8: ++tn4; break;
								case 9: ++tn1; ++tn4; break;
								case 10: ++tn2; ++tn4; break;
								case 11: ++tn1; ++tn2; ++tn4; break;
								case 12: ++tn3; ++tn4; ++tn6; break;
								case 13: ++tn1; ++tn3; ++tn4; ++tn6; break;
								case 14: ++tn2; ++tn3; ++tn4; ++tn6; break;
								case 15: ++tn1; ++tn2; ++tn3; ++tn4; ++tn6; break;
								case 0: ++tn7; break;
							}
						}
								
					} // jx
				} // jy
			} // jz
			
			cproxit = cproxit/(Nxyz);
			csub2t = csub2t/(Nxyz);
			csub3t = csub3t/(Nxyz);
			cjsrt = cjsrt/(Nxyz);
			cjt = cjt/(Nxyz);
			cnsrt = cnsrt/(Nxyz);
			cit /= (Nxyz);
			poto = poto/(Nxyz);
			pbcto = pbcto/(Nxyz)/100.0;

			isit = isit/(Nxyz);
			catft = catft/(Nxyz);
			catst = catst/(Nxyz);
			casrt = casrt/(Nxyz);
			camyot /= (Nxyz);
			mgmyot /= (Nxyz);
			cacalt = cacalt/(Nxyz);
			cadyet = cadyet/(Nxyz);
			leakt = leakt/(Nxyz);
			upt = upt/(Nxyz);
			ire = ire/(Nxyz);
			ncxflux /= (Nxyz);
			ncxfwd /= (Nxyz);
			icaflux /= (Nxyz);
			outAncx /= (Nxyz);

			casart /= (Nxyz);
			casarht /= (Nxyz);
			casarjt /= (Nxyz);
			casarhjt /= (Nxyz);

			////////////////////////////////////////////////////////////////////////////////
			/////////////////////////// check Ca2+ conservation ////////////////////////////
			TotalCa = (cit+ catft + catst + casrt + camyot + cacalt )*Vi*8 +
					(csubt + casart + casarht )*Vs +
					(cproxit + casarjt + casarhjt	)*Vp +
					cjt*Vjsr +
					cnsrt*Vnsr*8;
			
			////////////////////////////// output to screen /////////////////////////////
			end_time=clock()/(1.0*CLOCKS_PER_SEC);	
			printf(	"t=%f/%5.1f\tcit=%f\t\ttime=%6.1fs=%4.1fh\n",
					t, nbeat*PCL, 
					cit, end_time-start_time,
					(end_time-start_time)/3600.0
				  );
			///////////////////////////////////////////////////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////flag
			fprintf(wholecell_file,  "%f %f %f %f %f  %f %f %f %f %f  "
									"%f %f %f %f %f  %f %f %f %f %f  "
									"%f %f %f %f %f  %f %f %f %f %f  "
									"%f %f %f %f %f  %f %f %f %f %f  "
									"%i %f %f %f %f  %f %f %f\n",
									
									t, cit,
									v, xinacato,
									out_ica/(outt/DT), cproxit,
									csubt, cjsrt,
									cnsrt, poto,

									xnai, xiks,
									xikr, xik1,
									xinak, xitos,
									xitof, out_ina/(outt/DT),
									xr, ncxfwd*(Vs/Vp), 

									pbcto, cacalt, 
									catft, leakt, 
									upt, ire,
									tnou/(1.0*Nxyz), tnob/(1.0*Nxyz),
									tncu/(1.0*Nxyz), tncb/(1.0*Nxyz),

									tn1/(1.0*Nxyz), tn2/(1.0*Nxyz),
									tn3/(1.0*Nxyz), tn4/(1.0*Nxyz),
									tn5/(1.0*Nxyz), tn6/(1.0*Nxyz),
									tn7/(1.0*Nxyz), outAncx,
									xs1, qks,

									sparksum, ncxflux*(Vs/Vp),
									icaflux, casarjt,
									casarhjt, TotalCa,
									TotalCa - TotalCa_before, CaExt
					);
			fflush( wholecell_file );

			sparksum = 0;
			out_ica = 0;
			out_ina = 0;
			TotalCa_before=TotalCa;
			CaExt = 0;


			////////////////////////////////////////////////////////////////////////////				
			/////////////////////////////////// cs NCX ///////////////////////////////// for Fig. 5
			if ( step%2084800==0 && step>100 ) // at 52120 ms, at the peak of cs
			{
				for (jz=1;jz<nz-1;jz++)
				{
					for (jy=1;jy<ny-1;jy++)
					{
						for (jx=1;jx<nx-1;jx++)
						{
							fprintf(output_csNCX20, "%f\t%f\t%f\t%f\t%i\t"  "%i\t%i\t%i\n", 
													cs[pos(jx,jy,jz)],   CRU[pos(jx,jy,jz)].xinaca,
													CRU2[pos(jx,jy,jz)].cp, CRU2[pos(jx,jy,jz)].cjsr,
													CRU2[pos(jx,jy,jz)].ncu, CRU2[pos(jx,jy,jz)].ncb, 
													CRU2[pos(jx,jy,jz)].nou, CRU2[pos(jx,jy,jz)].nob
											 );
						}
					}
				}

			}
			fflush(output_csNCX20);


			if ( step%2096900==0 && step>100 ) // at 52422.5 ms, 
			{
				for (jz=1;jz<nz-1;jz++)
				{
					for (jy=1;jy<ny-1;jy++)
					{
						for (jx=1;jx<nx-1;jx++)
						{
							fprintf(output_csNCX322, "%f\t%f\t%f\t%f\t%i\t"  "%i\t%i\t%i\n", 
													cs[pos(jx,jy,jz)],   CRU[pos(jx,jy,jz)].xinaca,
													CRU2[pos(jx,jy,jz)].cp, CRU2[pos(jx,jy,jz)].cjsr,
													CRU2[pos(jx,jy,jz)].ncu, CRU2[pos(jx,jy,jz)].ncb, 
													CRU2[pos(jx,jy,jz)].nou, CRU2[pos(jx,jy,jz)].nob
									);
						}
					}
				}

			}
			fflush(output_csNCX322);


			if ( step%2098000==0 && step>100 ) // at 52450 ms, EAD onset point for the hyperacitve RyR model
			{
				for (jz=1;jz<nz-1;jz++)
				{
					for (jy=1;jy<ny-1;jy++)
					{
						for (jx=1;jx<nx-1;jx++)
						{
							fprintf(output_csNCX350,"%f\t%f\t%f\t%f\t%i\t"  "%i\t%i\t%i\n", 
													cs[pos(jx,jy,jz)],   CRU[pos(jx,jy,jz)].xinaca,
													CRU2[pos(jx,jy,jz)].cp, CRU2[pos(jx,jy,jz)].cjsr,
													CRU2[pos(jx,jy,jz)].ncu, CRU2[pos(jx,jy,jz)].ncb, 
													CRU2[pos(jx,jy,jz)].nou, CRU2[pos(jx,jy,jz)].nob
											 );
						}
					}
				}

			}
			fflush(output_csNCX350);
			////////////////////////////////////////////////////////////////////////////				
			/////////////////////////////////// Line Scan //////////////////////////////
			#ifdef outputlinescan
				if (t>40000)
				{
					for (jx =1; jx < nx-1; jx++)
					{
						int jz = 5;
						int jy = ny/2;
						fprintf(linescan_file, "%f %f %f %f %f ""%f %f %f %i %f "
											"%i %i %i %i %f ""%f %f %f \n",

											t,	(double)jx,
											ci[pos(jx,jy,jz)*8], cs[pos(jx,jy,jz)], 
											CRU2[pos(jx,jy,jz)].cp, CRU2[pos(jx,jy,jz)].cjsr,
											CRU[pos(jx,jy,jz)].xinaca, CRU2[pos(jx,jy,jz)].xire, 
											CRU2[pos(jx,jy,jz)].nl, CRU2[pos(jx,jy,jz)].Ancx, 

											CRU2[pos(jx,jy,jz)].nou, CRU2[pos(jx,jy,jz)].nob, 
											CRU2[pos(jx,jy,jz)].ncu, CRU2[pos(jx,jy,jz)].ncb,
											CRU2[pos(jx,jy,jz)].cjsr, CRU2[pos(jx-1,jy,jz)].cjsr,
											CRU2[pos(jx,jy+1,jz)].cjsr, CRU2[pos(jx,jy,jz-1)].cjsr
											
								);
					}
					fprintf(linescan_file, "\n");
					fflush(linescan_file);
				}
				
			#endif

			
		}
		step++;
		t=step*DT;
	
	}
	
	fclose(wholecell_file);

	#ifdef outputlinescan
		fclose(linescan_file);
	#endif

	fclose(output_csNCX20);
	fclose(output_csNCX322);
	fclose(output_csNCX350);

	free(CYT);
	free(CRU);
	free(CRU2);
	free(SBU);
	free(CBU);
	free(ci);
	free(cinext);
	free(cnsr);
	free(cnsrnext);
	free(cs);
	free(csnext);
	
	return EXIT_SUCCESS;	
}