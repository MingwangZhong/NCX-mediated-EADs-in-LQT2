// This file includes two functions: Initialize, Compute

// initialization 
void Initialize( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, double *cnsr, double *cnsrnext, 
						double *cs, double *csnext, double cjsr_b)
{
	int i,j,k,ps;
	for (k=1; k<nz-1; k++)
	{
		for (j=1; j<ny-1; j++)
		{
			for (i=1; i<nx-1; i++)
			{
				ps = pos(i,j,k);

				CRU2[ps].randomi = 1.0;
				CRU2[ps].cp = ci_basal;
				cs[ps] = ci_basal;
				csnext[ps] = ci_basal;
				CRU2[ps].cjsr = cjsr_b;
				CRU2[ps].nspark = 0;
				CRU2[ps].Ancx = 0.025;

				double ratio;
				for ( int ii = 0; ii < 8; ++ii )
				{
					ci[ps*8+ii] = ci_basal;
					cnsr[ps*8+ii] = cjsr_b;
					cinext[ps*8+ii] = ci_basal;
					cnsrnext[ps*8+ii] = cjsr_b;

					CBU[ps*8+ii].catf = ktfon*ci_basal*Btf/(ktfon*ci_basal+ktfoff);
					CBU[ps*8+ii].cats = ktson*ci_basal*Bts/(ktson*ci_basal+ktsoff);
					CBU[ps*8+ii].cacal = kcalon*ci_basal*Bcal/(kcalon*ci_basal+kcaloff);
					CBU[ps*8+ii].cadye = kdyeon*ci_basal*Bdye/(kdyeon*ci_basal+kdyeoff);
					CBU[ps*8+ii].casr = ksron*ci_basal*Bsr/(ksron*ci_basal+ksroff);
					
					ratio = Mgi*Kmyoca/(ci_basal*Kmyomg);
					CBU[ps*8+ii].camyo = ci_basal*Bmyo/(Kmyoca+ci_basal*(ratio+1.0));
					CBU[ps*8+ii].mgmyo = CBU[ps*8+ii].camyo*ratio;
				}
				SBU[ps].casar = ksaron*ci_basal*Bsar/(ksaron*ci_basal+ksaroff);
				SBU[ps].casarh = ksarhon*ci_basal*Bsarh/(ksarhon*ci_basal+ksarhoff);
				SBU[ps].casarj = ksaron*ci_basal*Bsar/(ksaron*ci_basal+ksaroff);
				SBU[ps].casarhj = ksarhon*ci_basal*Bsarh/(ksarhon*ci_basal+ksarhoff);
				
				int randpoint = svncp;
				
				for(int ll=0; ll<8; ll++)
				{
					if ( ll < randpoint )
						CRU2[ps].lcc[ll] = 3; // 4 LTCCs are in state 3
					else
						CRU2[ps].lcc[ll] = 16; // others are not used
				}

				
				double roo2 = ratedimer/(1.0+pow(kdimer/(cjsr_b),hilldimer));
				double kub = (-1.0+sqrt(1.0+8.0*BCSQN*roo2))/(4.0*roo2*BCSQN)/taub;
				double kbu = 1.0/tauu;

				double fracbound = 1/(1+kbu/kub);

				CRU2[ps].ncb = (int)(fracbound*nryr);
				CRU2[ps].ncu = nryr-(int)(fracbound*nryr);
				CRU2[ps].nob = 0;
				CRU2[ps].nou = 0;
			}
		}
	}

}


#define FINESTEP 5 // more steps for the proximal and submembrane spaces
#define DTF (DT/FINESTEP)	
void Compute( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, double *cnsr, double *cnsrnext, 
						double *cs, double *csnext, double v, int step, double Ku, double Kb, double cpstar, double xnai, 
						double sv_ica, double sv_ncx)
{
	int i,j,k,ps;
	for (k=1; k<nz-1; k++)
	{
		for (j=1; j<ny-1; j++)
		{
			for (i=1; i<nx-1; i++)
			{
				ps = pos(i,j,k);

				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////// ica /////////////////////////////////////
				if ( step%OUT_STEP==1 ) // just for output
				{
					CRU2[ps].nl=0;
					CRU2[ps].nspark=0;
				}

				double ICa = 0;
				#ifndef permeabilized
					int jj, ll, nlcp = 0;
					for (ll=0; CRU2[ps].lcc[ll]<16 && ll < 8; ll++)
					{
						jj = LTCC_gating(CRU2[ps].lcc[ll], (CRU2[ps].cp+cs[ps]*icagamma)/(1+icagamma), v);
						CRU2[ps].lcc[ll] = jj;
						if (jj==0 ) // LTCCs are in the open state
						{
							++CRU2[ps].nl;
							++nlcp;
						}
					}
					ICa = sv_ica*(double)(nlcp)*single_LTCC_current(v,(CRU2[ps].cp+cs[ps]*icagamma)/(1+icagamma))/(1+icagamma);
				#endif

				CRU[ps].xica = ICa;
				
				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////// RyR /////////////////////////////////////
				int presp = CRU2[ps].nou+CRU2[ps].nob;
				int problem = RyR_gating(Ku, Kb, cpstar, CRU2[ps].cp, CRU2[ps].cjsr, &CRU2[ps].ncu, &CRU2[ps].nou, 
										 &CRU2[ps].ncb, &CRU2[ps].nob, i, j, k );
				
				CRU2[ps].po = (double)(CRU2[ps].nou+CRU2[ps].nob)/(double)(nryr);
				CRU2[ps].xire = 0.0147*svjmax*CRU2[ps].po*(CRU2[ps].cjsr-CRU2[ps].cp)/Vp/CRU2[ps].randomi;
				if	(CRU2[ps].xire<0) 	CRU2[ps].xire = 0;
				
				if( presp < 3 && (CRU2[ps].nou+CRU2[ps].nob) >= 3 ) // to detect spark rate
			  		CRU2[ps].nspark = 1;
				///////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////// inaca /////////////////////////////////////
				#ifndef permeabilized
					CRU[ps].xinaca = sv_ncx*NCX( cs[ps]/1000.0, v, PCL, xnai, &CRU2[ps].Ancx );
				#else
					CRU[ps].xinaca = 0;
				#endif
				
				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////
				double dotcjsr; // dcjsr/dt
				double dotci[8]; // dci/dt
				double dotcnsr[8]; // dcnsr/dt
				double xicoupn[8]; // coupling of cnsr
				double xicoupi[8]; // coupling of ci
				
				double diffjn0 = (CRU2[ps].cjsr-cnsr[ps*8])/tautr/2.;
				double diffjn1 = (CRU2[ps].cjsr-cnsr[ps*8+1])/tautr/2.;

				for ( int ii = 0; ii < 8; ++ii )
				{
					int psi = ps*8+ii;

					CYT[psi].xiup = sviup*Uptake(ci[psi],cnsr[psi]);
					CYT[psi].xileak = svileak*0.00001035*(cnsr[psi]-ci[psi])/(1.0+pow2(500.0/cnsr[psi]));

					// For the index of 8 compartments, 
					// see Fig. S5: Terentyev et al (2014), Circulation research, 115(11), 919-928
					int north = (ii%2)?(pos(i,j,k+1)*8+ii-1):(psi+1);
					int south = (ii%2)?(psi-1):(pos(i,j,k-1)*8+ii+1);
					int east = ((ii/2)%2)?(pos(i,j+1,k)*8+ii-2):(psi+2);
					int west = ((ii/2)%2)?(psi-2):(pos(i,j-1,k)*8+ii+2);
					int top = ((ii/4)%2)?(pos(i+1,j,k)*8+ii-4):(psi+4);
					int bottom = ((ii/4)%2)?(psi-4):(pos(i-1,j,k)*8+ii+4);

					xicoupn[ii] =	(cnsr[north]-cnsr[psi])/(taunt) +
									(cnsr[south]-cnsr[psi])/(taunt) +
									(cnsr[east]-cnsr[psi])/(taunt) +
									(cnsr[west]-cnsr[psi])/(taunt) +
									(cnsr[top]-cnsr[psi])/(taunl) +
									(cnsr[bottom]-cnsr[psi])/(taunl) ;

					xicoupi[ii] =	(ci[north]-ci[psi])/(tauit) +
									(ci[south]-ci[psi])/(tauit) +
									(ci[east]-ci[psi])/(tauit) +
									(ci[west]-ci[psi])/(tauit) +
									(ci[top]-ci[psi])/(tauil) +
									(ci[bottom]-ci[psi])/(tauil) ;

					dotcnsr[ii]=(CYT[psi].xiup-CYT[psi].xileak)*Vi/Vnsr+xicoupn[ii];
						
					double buffers =	Tropf(CBU[psi].catf, ci[psi], DT)+
										Trops(CBU[psi].cats, ci[psi], DT)+
										buCal(CBU[psi].cacal, ci[psi], DT)+
										buSR(CBU[psi].casr, ci[psi], DT)+
										MyoCa(CBU[psi].camyo, CBU[psi].mgmyo, ci[psi], DT) ;

					dotci[ii]=	- CYT[psi].xiup
								+ CYT[psi].xileak
								- buffers
								+ xicoupi[ii];
				}
				dotcnsr[0] += diffjn0*Vjsr/Vnsr;
				dotcnsr[1] += diffjn1*Vjsr/Vnsr;


				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////
				csnext[ps] = cs[ps];
				for( int iii = 0; iii < FINESTEP; ++iii )
				{

					double diffpi0 = (CRU2[ps].cp-ci[ps*8])/taupi/2.0;
					double diffpi1 = (CRU2[ps].cp-ci[ps*8+1])/taupi/2.0;
					double diffsi0 = (csnext[ps]-ci[ps*8])/tausi/2.0;
					double diffsi1 = (csnext[ps]-ci[ps*8+1])/tausi/2.0;

					dotci[0] += diffsi0*(Vs/Vi)/FINESTEP;
					dotci[1] += diffsi1*(Vs/Vi)/FINESTEP;
					dotci[0] += diffpi0*(Vp/Vi)/FINESTEP;
					dotci[1] += diffpi1*(Vp/Vi)/FINESTEP;


					double diffps = (CRU2[ps].cp-csnext[ps])/taups;
					////////////////////// submembrane: dotcs /////////////////////////
					double csbuff = buSar(SBU[ps].casar, csnext[ps], DTF) ;
					double csbuffh = buSarH(SBU[ps].casarh, csnext[ps], DTF)	;

					double csdiff = (cs[pos(i,j,k+1)]+cs[pos(i,j,k-1)]-2*cs[ps])/(taust); //4.

					double dotcs =    CRU[ps].xinaca
									+ Vp/Vs*diffps - diffsi0 - diffsi1 + csdiff
									- csbuff - csbuffh ;

					csnext[ps] += dotcs*DTF;
					SBU[ps].casar += csbuff*DTF;
					SBU[ps].casarh += csbuffh*DTF;
					
					if( SBU[ps].casar < 0 )		SBU[ps].casar =0;
					if( SBU[ps].casarh < 0 )	SBU[ps].casarh =0;

					////////////////////// proximal space: dotcp ////////////////////// 
					double cpbuff = buSarj(SBU[ps].casarj, CRU2[ps].cp, DTF);
					double cpbuffh = buSarHj(SBU[ps].casarhj, CRU2[ps].cp, DTF);

					double dotcp =  CRU2[ps].xire - ICa
									- diffps -diffpi0 -diffpi1 
									- cpbuff - cpbuffh ;

					CRU2[ps].cp += dotcp*DTF;
					SBU[ps].casarj += cpbuff*DTF;
					SBU[ps].casarhj += cpbuffh*DTF;
					
					if( SBU[ps].casarj < 0 )		SBU[ps].casarj =0;
					if( SBU[ps].casarhj < 0 )		SBU[ps].casarhj =0;
				}


					if ( csnext[ps]<0 ) 			csnext[ps]=1e-6;
					if ( CRU2[ps].cp < 0 )  		CRU2[ps].cp=0;

				////////////////////// cytosolic space ////////////////////// 
				for ( int ii = 0; ii < 8; ++ii )
				{
					int psi = ps*8+ii;
					
					cinext[psi]=ci[psi]+dotci[ii]*DT;
					cnsrnext[psi]=cnsr[psi]+dotcnsr[ii]*DT;

					CBU[psi].catf += Tropf(CBU[psi].catf,ci[psi], DT)*DT;
					CBU[psi].cats += Trops(CBU[psi].cats,ci[psi], DT)*DT;
					CBU[psi].cacal += buCal(CBU[psi].cacal,ci[psi], DT)*DT;
					CBU[psi].casr += buSR(CBU[psi].casr,ci[psi], DT)*DT;
					CBU[psi].camyo += MyoCa(CBU[psi].camyo,CBU[psi].mgmyo,ci[psi], DT)*DT;
					CBU[psi].mgmyo += MyoMg(CBU[psi].camyo,CBU[psi].mgmyo,ci[psi], DT)*DT;
					CBU[psi].cadye += buDye(CBU[psi].cadye,ci[psi], DT)*DT;
					
					if( cinext[psi]<0) 				cinext[psi]=0;
					if( cnsrnext[psi]<0) 			cnsrnext[psi]=0;
					if( CBU[psi].catf < 0 )			CBU[psi].catf =0;
					if( CBU[psi].cats < 0 )			CBU[psi].cats =0;
					if( CBU[psi].cacal < 0 )		CBU[psi].cacal =0;
					if( CBU[psi].casr < 0 )			CBU[psi].casr =0;
					if( CBU[psi].camyo < 0 )		CBU[psi].camyo =0;
					if( CBU[psi].mgmyo < 0 )		CBU[psi].mgmyo =0;
					if( CBU[psi].cadye < 0 )		CBU[psi].cadye =0;
				
				}	
				

				dotcjsr = 1.0/(1.0 + (BCSQN*kbers*nCa)/pow2((kbers+CRU2[ps].cjsr)) )*( -diffjn0-diffjn1 -CRU2[ps].xire*Vp/Vjsr*CRU2[ps].randomi);

				CRU2[ps].cjsr += dotcjsr*DT;
				CRU2[ps].Tcj = CRU2[ps].cjsr + BCSQN*nCa*CRU2[ps].cjsr/(kbers+CRU2[ps].cjsr);

				
				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////
				//Boundary conditions

				#ifndef permeabilized
					if(k==1)
					{
						for( int ii = 1; ii < 8; ii+=2 ){
							cinext[pos(i,j,0)*8+ii]=cinext[pos(i,j,1)*8+ii-1];
							cnsrnext[pos(i,j,0)*8+ii]=cnsrnext[pos(i,j,1)*8+ii-1];
						}
						csnext[pos(i,j,0)]=csnext[pos(i,j,1)];
					}
					if(k==nz-2)
					{
						for( int ii = 0; ii < 8; ii+=2 ){
							cinext[pos(i,j,nz-1)*8+ii]=cinext[pos(i,j,nz-2)*8+ii+1];
							cnsrnext[pos(i,j,nz-1)*8+ii]=cnsrnext[pos(i,j,nz-2)*8+ii+1];
						}
						csnext[pos(i,j,nz-1)]=csnext[pos(i,j,nz-2)];
					}
					if(j==1)
					{
						for( int ii = 2; ii < 8; ii+=4 ){
							cinext[pos(i,0,k)*8+ii]=cinext[pos(i,1,k)*8+ii-2];
							cinext[pos(i,0,k)*8+ii+1]=cinext[pos(i,1,k)*8+ii-1];
							cnsrnext[pos(i,0,k)*8+ii]=cnsrnext[pos(i,1,k)*8+ii-2];
							cnsrnext[pos(i,0,k)*8+ii+1]=cnsrnext[pos(i,1,k)*8+ii-1];
						}
					}
					if(j==ny-2)
					{
						for( int ii = 0; ii < 8; ii+=4 ){
							cinext[pos(i,ny-1,k)*8+ii]=cinext[pos(i,ny-2,k)*8+ii+2];
							cinext[pos(i,ny-1,k)*8+ii+1]=cinext[pos(i,ny-2,k)*8+ii+3];
							cnsrnext[pos(i,ny-1,k)*8+ii]=cnsrnext[pos(i,ny-2,k)*8+ii+2];
							cnsrnext[pos(i,ny-1,k)*8+ii+1]=cnsrnext[pos(i,ny-2,k)*8+ii+3];
						}
					}
					if(i==1)
					{
						for( int ii = 4; ii < 8; ++ii ){
							cinext[pos(0,j,k)*8+ii]=cinext[pos(1,j,k)*8+ii-4];
							cnsrnext[pos(0,j,k)*8+ii]=cnsrnext[pos(1,j,k)*8+ii-4];
						}
					}
					if(i==nx-2)
					{
						for( int ii = 0; ii < 4; ++ii ){
							cinext[pos(nx-1,j,k)*8+ii]=cinext[pos(nx-2,j,k)*8+ii+4];
							cnsrnext[pos(nx-1,j,k)*8+ii]=cnsrnext[pos(nx-2,j,k)*8+ii+4];
						}
					}
				#else
					if(k==1)
						for( int ii = 1; ii < 8; ii+=2 )
							cnsrnext[pos(i,j,0)*8+ii]=cnsrnext[pos(i,j,1)*8+ii-1];
					if(k==nz-2)
						for( int ii = 0; ii < 8; ii+=2 )
							cnsrnext[pos(i,j,nz-1)*8+ii]=cnsrnext[pos(i,j,nz-2)*8+ii+1];
					if(j==1)
					{
						for( int ii = 2; ii < 8; ii+=4 ){
							cnsrnext[pos(i,0,k)*8+ii]=cnsrnext[pos(i,1,k)*8+ii-2];
							cnsrnext[pos(i,0,k)*8+ii+1]=cnsrnext[pos(i,1,k)*8+ii-1];
						}
					}
					if(j==ny-2)
					{
						for( int ii = 0; ii < 8; ii+=4 ){
							cnsrnext[pos(i,ny-1,k)*8+ii]=cnsrnext[pos(i,ny-2,k)*8+ii+2];
							cnsrnext[pos(i,ny-1,k)*8+ii+1]=cnsrnext[pos(i,ny-2,k)*8+ii+3];
						}
					}
					if(i==1)
						for( int ii = 4; ii < 8; ++ii )
							cnsrnext[pos(0,j,k)*8+ii]=cnsrnext[pos(1,j,k)*8+ii-4];
					if(i==nx-2)
						for( int ii = 0; ii < 4; ++ii )
							cnsrnext[pos(nx-1,j,k)*8+ii]=cnsrnext[pos(nx-2,j,k)*8+ii+4];
				#endif
			}
		}
	}
}