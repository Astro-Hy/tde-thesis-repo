C FILE: PROFILEell.F
      subroutine ellprofile(maxstep,npix,       
     &     xi1,xi2,broad,q1,q2,xib,angi,   
     &     ell,smooth,phi0,               
     &     nstep,olambda,                
     &     wave,flux)              
C----------------------------------------------------------------------------
C Calculates the profile of an emission lines from a circular disk with
C a pattern in the brightness distribution and considering radiative
C transfer through the base of an outflowing wind.
C============================================================================
      INTEGER maxstep,nstep
      INTEGER npix
      real*4 xi1,xi2,broad,q1,q2,xib
      real*4 angi,phi0,olambda
      REAL*4 WAVE(npix)
      REAL*8 flux(npix)     
      CHARACTER*1 SMOOTH
      REAL*8 PHIP(MAXSTEP)
      REAL*8 ELLMAX,ARG,ELLSLOPE,B_R 
      REAL*8 CNU,DOPPLER,BOOST,BEND,ELL
      real*8 xi,xistep,beta,ogamma,psi
      real*8 EXPON,XIPHIP,XIPHIPSTEP,XIPOW
      REAL*8 SINPHIP(MAXSTEP),COSPHIP(MAXSTEP)
      REAL*8 ELLSIN(MAXSTEP),ELLCOS(MAXSTEP)
      REAL*8 SINCOS(MAXSTEP),SINSIN(MAXSTEP)
      REAL*4 EX


Cf2py intent(in) maxstep
Cf2py intent(in) npix
Cf2py intent(in) xi1
Cf2py intent(in) xi2
Cf2py intent(in) broad
Cf2py intent(in) q1
Cf2py intent(in) q2
Cf2py intent(in) xib     
Cf2py intent(in) angi     
Cf2py intent(in) ell   
Cf2py intent(in) smooth
Cf2py intent(in) phi0
Cf2py intent(in) nstep
Cf2py intent(in) olambda
Cf2py depend(PHIP) maxstep
Cf2py depend(SINPHIP) maxstep
Cf2py depend(COSPHIP) maxstep
Cf2py depend(ELLSIN) maxstep
Cf2py depend(ELLCOSPHIP) maxstep
Cf2py depend(SINCOS) maxstep
Cf2py depend(SINSIN) maxstep
Cf2py intent(out) flux
Cf2py intent(in) wave
Cf2py depend(flux) npix
Cf2py depend(wave) npix
 
C--   Set the values of useful constants
      PI=3.14159
      clight=2.9979e5
      deg_per_rad=180./pi
      
C--   Translate imput parameters to proper units

C     Normalization constant for outer power law
      if (xib.le.xi1.or.xib.gt.xi2) then
         pownorm=1.
      else
         pownorm=xib**(q2-q1)
      endif
      
C     From i(degrees) to sin i, cos i
      SINI=SIN(ANGI/deg_per_rad)
      COSI=COS(ANGI/deg_per_rad)

C     Broadening parameter from km/s to v/c
      BROAD=BROAD/clight  !translate km/s to v/c

C     If eccentricity is smoothly varying, find the slope of e(r).
      
      IF (ELL.EQ.0.) SMOOTH='n'
      IF (SMOOTH.EQ.'y') THEN
         ELLSLOPE = 1.0
      ENDIF
      IF (SMOOTH.EQ.'y') THEN
         ELLMAX=ELL
         ELLSLOPE=ELLMAX/XI2
      ENDIF
      
C     Major axis orientation: from degrees to radians
      PHI0=PHI0/deg_per_rad

C     Make sure the max number of integration steps is not exceeded
      IF (NSTEP.GT.MAXSTEP) NSTEP=MAXSTEP

C--> Construct the arrays of trigonometric functions. These will be very
C    useful later in the CPU intensive loops.

      XIDEL = ALOG10(XI2/XI1)/NSTEP/2. ! log steps in radius
      XIDEL = 10.**XIDEL               ! 1/2 radial step size
      XIDIFF = (XIDEL-1./XIDEL)
      PHISTEP = 2.*PI/NSTEP      ! phi step size

      DO I=1,NSTEP
         PHIP(I)=0.5*PHISTEP*(2*I-1)
         SINPHIP(I)=SIN(PHIP(I))  ! phi grid pts
         COSPHIP(I)=COS(PHIP(I))
         SINCOS(I)=SINI*COSPHIP(I)
         SINSIN(I)=SINI*SINPHIP(I)
         ELLSIN(I)=ELL*SIN(PHIP(I)-PHI0)    ! trig funcs for constant e 
         ELLCOS(I)=1.-ELL*COS(PHIP(I)-PHI0)
      ENDDO
      
C--> This is the heart of the program. Three nested loops compute the line
C    profile as a function of wavelength, by integrating over the surface of 
C    the disk.
      
      
      DO K=1,NPIX
         EX=(OLAMBDA/WAVE(K))-1
         
         CNU=0.
         
         DO J=1,NSTEP
            
            XI=XI1*XIDEL**(2*J-1)
            XISTEP=XI*XIDIFF
            IF (SMOOTH.EQ.'y') ELL=ELLSLOPE*XI

            DO I=1,NSTEP

               IF (SMOOTH.EQ.'y') THEN ! trig funcs for varying e 
                  ELLSIN(I)=ELL*SIN(PHIP(I)-PHI0)
                  ELLCOS(I)=1.-ELL*COS(PHIP(I)-PHI0)
               ENDIF

               XIPHIP=XI*(1+ELL)/ELLCOS(I)
               XIPHIPSTEP=XISTEP*(1+ELL)/ELLCOS(I)

c               ALPHA=1-(3./XIPHIP)
               BETA=1-(2./XIPHIP)
               OGAMMA=BETA*BETA-((ELLSIN(I)*ELLSIN(I)/ELLCOS(I))
     &               +BETA*ELLCOS(I))/XIPHIP
               OGAMMA=BETA/SQRT(OGAMMA)
               PSI=1.+((1.-SINCOS(I))/(1.+SINCOS(I)))/XIPHIP
               B_R=SQRT(1.-SINCOS(I)*SINCOS(I))*PSI
               if (b_r.gt.1.) b_r=1.
               DOPPLER=(OGAMMA/BETA)*(SQRT(BETA)
     &                -(ELLSIN(I)/SQRT(XIPHIP*ELLCOS(I)))
     &                     *SQRT((1./BETA)-B_R*B_R)
     &                +SQRT(ELLCOS(I)*BETA/XIPHIP)*SINSIN(I)*PSI)
               
               DOPPLER=1./DOPPLER
               BOOST=DOPPLER*DOPPLER*DOPPLER
               EXPON=(1.+EX-DOPPLER)/DOPPLER/BROAD
               EXPON=EXPON*EXPON/2.
               BEND=(1+((1-SINCOS(I))/(1+SINCOS(I)))/XIPHIP)
               if (xi.lt.xib) then
                  XIPOW=XI**(1-Q1)
               else
                  XIPOW=POWNORM*XI**(1-Q2)
               endif
               
                ARG=
     &              BOOST
     &              *XIPOW
     &              *BEND
     &              *EXP(-EXPON)
               CNU=CNU+ARG*XIPHIPSTEP*PHISTEP
            ENDDO
         ENDDO
         FLUX(K)=CNU
      ENDDO
      

      RETURN
      END
C END FILE PROFILEELL.F
