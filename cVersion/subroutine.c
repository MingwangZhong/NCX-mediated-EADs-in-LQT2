// This file includes the functions called by Compute

// Buffering parameters are mainly from the book:
// DM Bers, 2002, Cardiac excitation-contraction coupling
#define ktson 0.00254
#define ktsoff 0.000033
#define Bts 134.0

#define ktfon 0.0327
#define ktfoff 0.0196
#define Btf 70.0

#define kcalon 0.0543
#define kcaloff 0.238
#define Bcal 24.0

#define kdyeon 0.256
#define kdyeoff 0.1
#define Bdye 50.0

#define ksron 0.1
#define ksroff 0.06
#define Bsr 19.0

// The total amount of sarcolemma buffer is in unit of (uM/l cyt)
#define ksaron 0.1
#define ksaroff 1.3
#define Bsar (42*(Vi*8.0/Vs)*1.2)
#define KSAR (ksaroff/ksaron)

#define ksarhon 0.1
#define ksarhoff 0.03
#define Bsarh (15.0*(Vi*8.0/Vs)*1.2)
#define KSARH (ksarhoff/ksarhon)


#define Bmyo 	140
#define konmyomg 	0.0000157
#define koffmyomg 	0.000057
#define konmyoca 	0.0138
#define koffmyoca 	0.00046
#define Mgi 	500
#define Kmyomg 	(koffmyomg/konmyomg)
#define Kmyoca 	(koffmyoca/konmyoca)

double MyoCa(double CaMyo, double MgMyo, double calciu, double dt)
{
	double Itc;
	if( konmyoca*calciu*dt > 1 )
		Itc=0.95/dt*(Bmyo-CaMyo-MgMyo)-koffmyoca*CaMyo;
	else
		Itc=konmyoca*calciu*(Bmyo-CaMyo-MgMyo)-koffmyoca*CaMyo;
	return(Itc);
}

double MyoMg(double CaMyo, double MgMyo, double calciu, double dt)
{
	double Itc;
	if( konmyomg*Mgi*dt > 1 )
		Itc=0.95/dt*(Bmyo-CaMyo-MgMyo)-koffmyomg*MgMyo;
	else
		Itc=konmyomg*Mgi*(Bmyo-CaMyo-MgMyo)-koffmyomg*MgMyo;
	return(Itc);
}

double Tropf(double CaTf, double calciu, double dt)
{
	double Itc;
	if( ktfon*calciu*dt > 1 )
		Itc=0.95/dt*(Btf-CaTf)-ktfoff*CaTf;
	else
		Itc=ktfon*calciu*(Btf-CaTf)-ktfoff*CaTf;
	return(Itc);
}

double Trops(double CaTs, double calciu, double dt)
{
	double Itc;
	if( ktson*calciu*dt > 1 )
		Itc=0.95/dt*(Bts-CaTs)-ktsoff*CaTs;
	else
		Itc=ktson*calciu*(Bts-CaTs)-ktsoff*CaTs;
	return(Itc);
}

double buCal(double CaCal, double calciu, double dt)
{
	double Itc;
	if( kcalon*calciu*dt > 1 )
		Itc=0.95/dt*(Bcal-CaCal)-kcaloff*CaCal;
	else
		Itc=kcalon*calciu*(Bcal-CaCal)-kcaloff*CaCal;
	return(Itc);
}

double buDye(double CaDye, double calciu, double dt)
{
	double Itc;
	if( kdyeon*calciu*dt > 1 )
		Itc=0.95/dt*(Bdye-CaDye)-kdyeoff*CaDye;
	else
		Itc=kdyeon*calciu*(Bdye-CaDye)-kdyeoff*CaDye;
	return(Itc);
}

double buSR(double CaSR, double calciu, double dt)
{
	double Itc;
	if( ksron*calciu*dt > 1 )
		Itc=0.95/dt*(Bsr-CaSR)-ksroff*CaSR;
	else
		Itc=ksron*calciu*(Bsr-CaSR)-ksroff*CaSR;
	return(Itc);
}

double buSar(double CaSar, double calciu, double dt)
{
	double Itc;
	if( ksaron*calciu*dt > 1 )
		Itc=0.95/dt*(Bsar-CaSar)-ksaroff*CaSar;
	else
		Itc=ksaron*calciu*(Bsar-CaSar)-ksaroff*CaSar;
	return(Itc);
}

double buSarH(double CaSarh, double calciu, double dt)
{
	double Itc;
	if( ksarhon*calciu*dt > 1 )
		Itc=0.95/dt*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	else
		Itc=ksarhon*calciu*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	return(Itc);
}

double buSarj(double CaSar, double calciu, double dt)
{
	double Itc;
	if( ksaron*calciu*dt > 1 )
		Itc=0.95/dt*(Bsar-CaSar)-ksaroff*CaSar;
	else
		Itc=ksaron*calciu*(Bsar-CaSar)-ksaroff*CaSar;
	return(Itc);
}

double buSarHj(double CaSarh, double calciu, double dt)
{
	double Itc;
	if( ksarhon*calciu*dt > 1 )
		Itc=0.95/dt*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	else
		Itc=ksarhon*calciu*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	return(Itc);
}





int LTCC_gating(int i, double cpx,double v)
{	
	double dv5 = 5;
	double dvk = 8;

	double fv5 = -22.8;
	double fvk = 9.1;

	double alphac = 0.22;
	double betac = 4;
	
	#ifdef ISO
		betac=2;
		dv5 = 0;
		fv5 = -28;
		fvk = 8.5;
	#endif

	double dinf = 1.0/(1.0+exp(-(v-dv5)/dvk));
	double taudin = 1.0/((1.0-exp(-(v-dv5)/dvk))/(0.035*(v-dv5))*dinf); // 1/taud
	if( (v < 0.0001) && (v > -0.0001) )
		taudin = 0.035*dvk/dinf;
	
	double finf = 1.0-1.0/(1.0+exp(-(v-fv5)/fvk))/(1.+exp((v-60)/12.0));
	double taufin = (0.02-0.007*exp(-pow2(0.0337*(v+10.5))));// 1/tauf
	
	double alphad = dinf*taudin;
	double betad = (1-dinf)*taudin;
	
	double alphaf = (finf)*taufin;
	double betaf = (1-finf)*taufin;
	
	double alphafca = 0.006;
	double betafca = 0.175/(1+pow2(35.0/cpx));
	
	double ragg=((double)rand()/RAND_MAX);
	double rig = ragg/DT;
	
	if ( (i%2) )
		if ( rig < alphac )
			return i-1;
		else
			rig-=alphac;
	else
		if ( rig < betac )
			return i+1;
		else
			rig-=betac;
		
	if ( ((i/2)%2) )
		if ( rig < alphad )
			return i-2;
		else
			rig-=alphad;
	else
		if ( rig < betad )
			return i+2;
		else
			rig-=betad;
	
	
	if ( ((i/4)%2) )
		if ( rig < alphaf )
			return i-4;
		else
			rig-=alphaf;
	else
		if ( rig < betaf )
			return i+4;
		else
			rig-=betaf;
	
	
	if ( ((i/8)%2) )
		if ( rig < alphafca )
			return i-8;
		else
			rig-=alphafca;
	else
		if ( rig < betafca )
			return i+8;
		else
			rig-=betafca;
	
	return(i);

	
}


double NCX(double cs, double v,double tperiod, double xnai, double *Ancx)	//Na_Ca exchanger
{
	double za=v*Farad/xR/Temper;
	double Ka = *Ancx;

	double t1=Kmcai*pow3(xnao)*(1.0+pow3(xnai/Kmnai));
	double t2=pow3(Kmnao)*cs*(1.0+cs/Kmcai);
	double t3=(Kmcao+Cext)*pow3(xnai)+cs*pow3(xnao);
	double Inaca=Ka*vnaca*(exp(eta*za)*pow3(xnai)*Cext-exp((eta-1.0)*za)*pow3(xnao)*cs)/((t1+t2+t3)*(1.0+ksat*exp((eta-1.0)*za)));
	
	double Ancxdot = (1.0/(1.0+pow3(0.0003/cs))-*Ancx)/150.;
	*Ancx += Ancxdot*DT;

	return (Inaca);
}

double Uptake(double ci, double cnsr)		//uptake
{
	double Iup;
	double vup=0.3;
	double Ki=0.123;
	double Knsr=1700.0;
	double HH=1.787;
	double upfactor=1.0;
	Iup = upfactor*vup*(pow(ci/Ki,HH)-pow(cnsr/Knsr,HH))/(1.0+pow(ci/Ki,HH)+pow(cnsr/Knsr,HH));
	return(Iup);
}


double single_LTCC_current(double v, double cp)	// single LTCC channel current
{
	double za=v*Farad/xR/Temper;
	double ica; 
		
	if (fabs(za)<0.001) 
	{
		ica=2.0*Pca*Farad*gammai*(cp/1000.0*exp(2.0*za)-Cext);
	}
	else 
	{
		ica=4.0*Pca*za*Farad*gammai*(cp/1000.0*exp(2.0*za)-Cext)/(exp(2.0*za)-1.0);
	}
	if (ica > 0.0)
		ica=0.0;
	return (ica);
}



int RyR_gating (double Ku, double Kb, double cpstar, double cp, double cjsr, int * ncu, int * nou, 
							int * ncb, int * nob, int i, int j, int k)
{
	double ku = Ku * 1.0/(1+pow2(5000/cjsr)) * 1.0/(1+pow(cpstar/cp,2)); // open rate from CU --> OU
	double kb = Kb * 1.0/(1+pow2(5000/cjsr)) * 1.0/(1+pow(cpstar/cp,2)); // open rate from CB --> OB

	double kuminus=1.0/taucu; // OU --> CU
	double kbminus=1.0/taucb; // OB --> CB

	if (ku*DT > 1.0) ku = 1.0/DT;
	if (kb*DT > 1.0) kb = 1.0/DT;
	if (kuminus*DT > 1.0) kuminus = 1.0/DT;
	if (kbminus*DT > 1.0) kbminus = 1.0/DT; 
 
	double kub=((1.0)/( 1.0+pow(31.*cjsr/13.3/(kbers+cjsr), 24) ))/taub; // OU --> OB,  CU --> CB
	double kbu=1.0/tauu; // CB --> CU

	double rate_ou_cu; // DT*kuminus
	double rate_cu_ou; // DT*ku
	double rate_ou_ob; // DT*kub
	double rate_ob_ou; // DT*kbu*ku/kb
	double rate_cb_cu; // DT*kbu
	double rate_cu_cb; // DT*kub
	double rate_ob_cb; // DT*kbminus
	double rate_cb_ob; // DT*kb

	int kk;
	double puu;
	double u1; // random number
	double u2; // random number
	double re; // random number

	int n_ou_cu;
	int n_cu_ou;
	int n_ou_ob;
	int n_ob_ou;
	int n_cu_cb;
	int n_cb_cu;
	int n_ob_cb;
	int n_cb_ob;
 
 
	int ret = 0; // not really used
	//////////////////////////////////// going from OU >> CU ///////////////////////////////
	rate_ou_cu = kuminus*DT;
	n_ou_cu=-1;
	if ((*nou) <=1 || rate_ou_cu < 0.2 || ((*nou) <=5 && rate_ou_cu < 0.3))
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*nou)*rate_ou_cu) && kk < 195)	//generates poisson number = fraction of closed RyR's that open
		{
			kk++;
			re=((double)rand()/RAND_MAX);
			puu=puu*(re);
		}
		n_ou_cu = kk-1;
	}
	else
	{
		kk = 0;
		while(n_ou_cu < 0)
		{
			//next is really a gaussian
			u1=((double)rand()/RAND_MAX);
			u2=((double)rand()/RAND_MAX);
			n_ou_cu = floor((*nou)*rate_ou_cu +sqrt((*nou)*rate_ou_cu*(1.0-rate_ou_cu))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ rand()%2;
			kk++;
			if( kk > 200)
			{
				n_ou_cu = 0;
				ret = 100000*i+1000*j+10*k+1;
			}
		}
	}
	if(n_ou_cu > nryr) {n_ou_cu = nryr;}
	
	//////////////////////////////////// going from CU >> OU ///////////////////////////////
	rate_cu_ou = ku*DT;
	n_cu_ou = -1;
	if((*ncu) <= 1 ||rate_cu_ou < 0.2 || ((*ncu) <= 5 && rate_cu_ou < 0.3))	//checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*ncu)*rate_cu_ou) && kk < 195)			//generates poisson number = fraction of closed RyR's that open
		{
			kk++;
			re=((double)rand()/RAND_MAX);
			puu=puu*re;
		}
		n_cu_ou = kk-1;
	}
	else
	{
		kk = 0;
		while(n_cu_ou < 0)
		{
			//next is really a gaussian
			u1=((double)rand()/RAND_MAX);
			u2=((double)rand()/RAND_MAX);
			n_cu_ou = floor((*ncu)*rate_cu_ou +sqrt((*ncu)*rate_cu_ou*(1.0-rate_cu_ou))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ rand()%2;
			kk++;
			if( kk > 200)
			{
				n_cu_ou = 0;
				ret = 100000*i+1000*j+10*k+2;
			}
		}
	}
	if(n_cu_ou > nryr) {n_cu_ou = nryr;}
	
		
	
	//////////////////////////////////// going from OU >> OB ///////////////////////////////
	rate_ou_ob = kub*DT;
	n_ou_ob = -1;
	if((*nou) <= 1 || rate_ou_ob < 0.2 || (*nou <= 5 && rate_ou_ob < 0.3)) //checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*nou)*rate_ou_ob) && kk < 195)			//generates poisson number = fraction of open RyR's that close
		{
			kk++;
			re=((double)rand()/RAND_MAX);
			puu=puu*re;
		}
		n_ou_ob = kk-1;
	}
	else
	{
		kk = 0;
		while(n_ou_ob < 0)
		{
			//next is really a gaussian
			u1=((double)rand()/RAND_MAX);
			u2=((double)rand()/RAND_MAX);
			n_ou_ob = floor(*nou*rate_ou_ob +sqrt(*nou*rate_ou_ob*(1.0-rate_ou_ob))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ rand()%2;
			kk++;
			if( kk > 200)
			{
				n_ou_ob = 0;
				ret = 100000*i+1000*j+10*k+3;
			}
		}
	}
	if(n_ou_ob > nryr) n_ou_ob = nryr;
		
	//////////////////////////////////// going from OB >> OU ///////////////////////////////
	rate_ob_ou = kbu*DT*(ku/kb);
	n_ob_ou = -1;
		
	if((*nob) <= 1 || rate_ob_ou < 0.2 || (*nob <= 5 && rate_ob_ou < 0.3)) //checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*nob)*rate_ob_ou) && kk < 195)			//generates poisson number = fraction of open RyR's that close
		{
			kk++;
			re=((double)rand()/RAND_MAX);
			puu=puu*re;
		}
		n_ob_ou = kk-1;
		
	}	
	else
	{
		kk = 0;
		while(n_ob_ou < 0)
		{		
			//next is really a gaussian
			u1=((double)rand()/RAND_MAX);
			u2=((double)rand()/RAND_MAX);
			n_ob_ou = floor(*nob*rate_ob_ou +sqrt(*nob*rate_ob_ou*(1.0-rate_ob_ou))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ rand()%2;
			kk++;
			if( kk > 200)
			{
				n_ob_ou = 0;
				ret = 100000*i+1000*j+10*k+4;
			}
		}
	}
	if(n_ob_ou > nryr) n_ob_ou = nryr;
	
		
		
	//////////////////////////////////// going from CB >> CU ///////////////////////////////
	rate_cb_cu = kbu*DT;
	n_cb_cu = -1;
	if((*ncb) <= 1 || rate_cb_cu < 0.2	|| (rate_cb_cu < 0.3 && (*ncb) <= 5))	//checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*ncb)*rate_cb_cu) && kk < 195)			//generates poisson number = fraction of open RyR's that close
		{
			kk++;
			re=((double)rand()/RAND_MAX);
			puu=puu*re;
		}
		n_cb_cu = kk-1;
	}
	else
	{
		kk = 0;
		while(n_cb_cu < 0)
		{		
			//next is really a gaussian
			u1=((double)rand()/RAND_MAX);
			u2=((double)rand()/RAND_MAX);
			n_cb_cu = floor((*ncb)*rate_cb_cu +sqrt((*ncb)*rate_cb_cu*(1.0-rate_cb_cu))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ rand()%2;
			kk++;
			if( kk > 200)
			{
				n_cb_cu = 0;
				ret = 100000*i+1000*j+10*k+5;
			}
		}
	}
	if(n_cb_cu > nryr) {n_cb_cu = nryr;}
	
	//////////////////////////////////// going from CU >> CB ///////////////////////////////
	rate_cu_cb = kub*DT;
	n_cu_cb = -1;
	if(*ncu <= 1 || rate_cu_cb < 0.2 || (*ncu <= 5 && rate_cu_cb < 0.3)) //checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-*ncu*rate_cu_cb) && kk < 195)			//generates poisson number = fraction of open RyR's that close
		{
			kk++;
			re=((double)rand()/RAND_MAX);
			puu=puu*re;
		}
		n_cu_cb = kk-1;
	}
	else
	{
		kk = 0;
		while(n_cu_cb < 0)
		{		
			//next is really a gaussian
			u1=((double)rand()/RAND_MAX);
			u2=((double)rand()/RAND_MAX);
			n_cu_cb = floor(*ncu*rate_cu_cb +sqrt(*ncu*rate_cu_cb*(1.0-rate_cu_cb))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ rand()%2;
			kk++;
			if( kk > 200)
			{
				n_cu_cb = 0;
				ret = 100000*i+1000*j+10*k+6;
			}
		}
	}
	if(n_cu_cb > nryr) {n_cu_cb = nryr;}
	

	
	//////////////////////////////////// going from OB >> CB ///////////////////////////////
	rate_ob_cb = kbminus*DT;
	n_ob_cb = -1;
	if((*nob) <= 1 || rate_ob_cb < 0.2 || ((*nob) <= 5 && rate_ob_cb < 0.3)) //checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*nob)*rate_ob_cb) && kk < 195)			//generates poisson number = fraction of closed RyR's that open
		{
			kk++;
			re=((double)rand()/RAND_MAX);
			puu=puu*re;
		}
		n_ob_cb = kk-1;
	}
	else
	{
		kk = 0;
		while(n_ob_cb < 0)
		{			
			//next is really a gaussian
			u1=((double)rand()/RAND_MAX);
			u2=((double)rand()/RAND_MAX);
			n_ob_cb = floor((*nob)*rate_ob_cb +sqrt((*nob)*rate_ob_cb*(1.0-rate_ob_cb))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ rand()%2;
			kk++;
			if( kk > 200)
			{
				n_ob_cb = 0;
				ret = 100000*i+1000*j+10*k+7;
			}
		}
	}
	//if(n_ob_cb > nn) {n_ob_cb = nn;}
	
	//////////////////////////////////// going from CB >> OB ///////////////////////////////
	rate_cb_ob = kb*DT;
	n_cb_ob = -1;
	if((*ncb) <= 1 || rate_cb_ob < 0.2 || ((*ncb) <= 5 && rate_cb_ob < 0.3)) //checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*ncb)*rate_cb_ob) && kk < 195)			//generates poisson number = fraction of closed RyR's that open
		{
			kk++;
			re=((double)rand()/RAND_MAX);
			puu=puu*re;
		}
		n_cb_ob = kk-1;
	}
	else
	{
		kk = 0;
		while(n_cb_ob < 0)
		{
			//next is really a gaussian
			u1=((double)rand()/RAND_MAX);
			u2=((double)rand()/RAND_MAX);
			n_cb_ob = floor((*ncb)*rate_cb_ob +sqrt((*ncb)*rate_cb_ob*(1.0-rate_cb_ob))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ rand()%2;
			kk++;
			if( kk > 200)
			{
				n_cb_ob = 0;
				ret = 100000*i+1000*j+10*k+8;
			}
		}
	}
	if(n_cb_ob > nryr) {n_cb_ob = nryr;}
	

	/////////////////////////////////////////////////////////////////////////////////////////
		
	if(n_ou_ob	+	n_ou_cu > *nou)
	{
		if(n_ou_cu >= n_ou_ob) n_ou_cu = 0;
		else	n_ou_ob = 0;
		if (n_ou_ob > *nou) n_ou_ob = 0;
		else if(n_ou_cu > *nou) n_ou_cu = 0;
	}
		
	if(n_ob_ou	+	n_ob_cb > *nob)
	{ 
		if(n_ob_ou >= n_ob_cb) n_ob_ou = 0;
		else	n_ob_cb = 0;
		if (n_ob_cb > *nob) n_ob_cb = 0;
		else if(n_ob_ou > *nob) n_ob_ou = 0;
	}
		
	if(n_cu_ou	+	n_cu_cb > *ncu ) 
	{
		if(n_cu_cb >= n_cu_ou) n_cu_cb = 0;
		else	n_cu_ou = 0;
		if (n_cu_ou > *ncu) n_cu_ou = 0;
		else if(n_cu_cb > *ncu) n_cu_cb = 0;
	}
		
		
	*nou += - n_ou_ob - n_ou_cu	+ n_ob_ou + n_cu_ou;
	if(*nou<0)		(*nou)=0;
	if(*nou>nryr)	*nou=nryr; 
	
	*nob += - n_ob_ou - n_ob_cb + n_ou_ob + n_cb_ob;
	if(*nob<0)			*nob=0;
	if(*nob>nryr)		*nob=nryr;
		
	*ncu += - n_cu_ou - n_cu_cb + n_ou_cu + n_cb_cu;
	if(*ncu<0) 			*ncu=0;
	if(*ncu>nryr)		*ncu=nryr;

	*ncb = nryr - *nou - *nob - *ncu;
	
	if(*ncb<0) 			*ncb=0;
	if(*ncb>nryr)		*ncb=nryr;

	return ret;
}