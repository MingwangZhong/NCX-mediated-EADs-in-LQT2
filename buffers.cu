// This file includes functions of buffers
// They are mainly from the book:
// DM Bers, 2002, Cardiac excitation-contraction coupling, 

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


__device__ double MyoCa(double CaMyo, double MgMyo, double calciu, double dt)
{
	double Itc;
	if( konmyoca*calciu*dt > 1 )
		Itc=0.95/dt*(Bmyo-CaMyo-MgMyo)-koffmyoca*CaMyo;
	else
		Itc=konmyoca*calciu*(Bmyo-CaMyo-MgMyo)-koffmyoca*CaMyo;
	return(Itc);
}


__device__ double MyoMg(double CaMyo, double MgMyo, double calciu, double dt)
{
	double Itc;
	if( konmyomg*Mgi*dt > 1 )
		Itc=0.95/dt*(Bmyo-CaMyo-MgMyo)-koffmyomg*MgMyo;
	else
		Itc=konmyomg*Mgi*(Bmyo-CaMyo-MgMyo)-koffmyomg*MgMyo;
	return(Itc);
}

__device__ double Tropf(double CaTf, double calciu, double dt)
{
	double Itc;
	if( ktfon*calciu*dt > 1 )
		Itc=0.95/dt*(Btf-CaTf)-ktfoff*CaTf;
	else
		Itc=ktfon*calciu*(Btf-CaTf)-ktfoff*CaTf;
	return(Itc);
}

__device__ double Trops(double CaTs, double calciu, double dt)
{
	double Itc;
	if( ktson*calciu*dt > 1 )
		Itc=0.95/dt*(Bts-CaTs)-ktsoff*CaTs;
	else
		Itc=ktson*calciu*(Bts-CaTs)-ktsoff*CaTs;
	return(Itc);
}

__device__ double buCal(double CaCal, double calciu, double dt)
{
	double Itc;
	if( kcalon*calciu*dt > 1 )
		Itc=0.95/dt*(Bcal-CaCal)-kcaloff*CaCal;
	else
		Itc=kcalon*calciu*(Bcal-CaCal)-kcaloff*CaCal;
	return(Itc);
}

__device__ double buDye(double CaDye, double calciu, double dt)
{
	double Itc;
	if( kdyeon*calciu*dt > 1 )
		Itc=0.95/dt*(Bdye-CaDye)-kdyeoff*CaDye;
	else
		Itc=kdyeon*calciu*(Bdye-CaDye)-kdyeoff*CaDye;
	return(Itc);
}

__device__ double buSR(double CaSR, double calciu, double dt)
{
	double Itc;
	if( ksron*calciu*dt > 1 )
		Itc=0.95/dt*(Bsr-CaSR)-ksroff*CaSR;
	else
		Itc=ksron*calciu*(Bsr-CaSR)-ksroff*CaSR;
	return(Itc);
}


__device__ double buSar(double CaSar, double calciu, double dt)
{
	double Itc;
	if( ksaron*calciu*dt > 1 )
		Itc=0.95/dt*(Bsar-CaSar)-ksaroff*CaSar;
	else
		Itc=ksaron*calciu*(Bsar-CaSar)-ksaroff*CaSar;
	return(Itc);
}


__device__ double buSarH(double CaSarh, double calciu, double dt)
{
	double Itc;
	if( ksarhon*calciu*dt > 1 )
		Itc=0.95/dt*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	else
		Itc=ksarhon*calciu*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	return(Itc);
}


__device__ double buSarj(double CaSar, double calciu, double dt)
{
	double Itc;
	if( ksaron*calciu*dt > 1 )
		Itc=0.95/dt*(Bsar-CaSar)-ksaroff*CaSar;
	else
		Itc=ksaron*calciu*(Bsar-CaSar)-ksaroff*CaSar;
	return(Itc);
}


__device__ double buSarHj(double CaSarh, double calciu, double dt)
{
	double Itc;
	if( ksarhon*calciu*dt > 1 )
		Itc=0.95/dt*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	else
		Itc=ksarhon*calciu*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	return(Itc);
}