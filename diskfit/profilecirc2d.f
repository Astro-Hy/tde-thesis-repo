C FILE: PROFILEcirc2d.F
      SUBROUTINE PROFILE(maxstep,npar, 
     &     xi1,xi2,broad,q1,q2,xib,angi, 
     &     anglam,t0,eta,version,       
     &     amp,narms,aobs,pitch,width,xispin,xispout,
     &     nstep,relativistic,olambda,              
     &     npix,wave,flux)
C
C     CALCULATE FIRST N FIBONACCI NUMBERS
C
      INTEGER maxstep, npar, nstep, npix
      INTEGER narms(npar)
      INTEGER P,I,K
      REAL*8 SINCOS(MAXSTEP),SINSIN(MAXSTEP)
      REAL*8 CNU,DOPPLER,BOOST,BEND,XX,ELEMENT
      real*8 sini,cosi,sinlam,cotlam
      real*8 omega,tau,direscprob,arg,radial
      real*8 EXPON
      real*4 xi1(npar),xi2(npar),broad(npar)
      real*4 q1(npar),q2(npar),xib(npar)
      real*4 xi
      real*4 angi(npar),anglam(npar),t0(npar)
      real*4 eta(npar),amp(npar)
      real*4 pitch(npar),width(npar)
      real*4 xispin(npar),xispout(npar)
      real*4 aobs(npar),olambda(npar)
      REAL*4 PHIP(MAXSTEP),WAVE(npix)
      REAL*8 flux(npix,npar)
      character*1 version,relativistic

Cf2py intent(in) maxstep
Cf2py intent(in) npar 
Cf2py intent(in) npix
Cf2py intent(in) xi1
Cf2py intent(in) xi2
Cf2py intent(in) broad
Cf2py intent(in) q1
Cf2py intent(in) q2
Cf2py intent(in) xib     
Cf2py intent(in) angi     
Cf2py intent(in) anglam   
Cf2py intent(in) t0
Cf2py intent(in) eta
Cf2py intent(in) version
Cf2py intent(in) amp
Cf2py intent(in) narms
Cf2py intent(in) aobs
Cf2py intent(in) pitch
Cf2py intent(in) width
Cf2py intent(in) xispin
Cf2py intent(in) xispout
Cf2py intent(in) nstep
Cf2py intent(in) relativistic
Cf2py intent(in) olambda
Cf2py intent(in) wave
Cf2py intent(out) flux
Cf2py depend(flux) npix
Cf2py depend(flux) npar
Cf2py depend(PHIP) maxstep
Cf2py depend(xi1) npar
Cf2py depend(xi2) npar
Cf2py depend(broad) npar
Cf2py depend(q1) npar
Cf2py depend(q2) npar
Cf2py depend(xib) npar
Cf2py depend(angi) npar
Cf2py depend(anglam) npar
Cf2py depend(t0) npar
Cf2py depend(eta) npar
Cf2py depend(amp) npar
Cf2py depend(pitch) npar
Cf2py depend(width) npar
Cf2py depend(xispin) npar
Cf2py depend(xispout) npar
Cf2py depend(aobs) npar
Cf2py depend(narms) npar      
Cf2py depend(olambda) npar

Cf2py depend(wave) npix

c      write(*,*) maxstep,                      ! array dimension
c     &     xi1,xi2,broad,q,angi,                       ! disk parameters
c     &     anglam,t0,eta,version,                      ! wind parameters
c     &     amp,narms,aobs,pitch,width,xispin,xispout,  ! spiral parameters
c     &     nstep,relativistic,olambda,                 ! computation settings
c     &     npix
      
C--   Set the values of useful constants
      
      cappa=1./4.7 ! ratio of Keplerian speed to terminal speed

      PI=3.14159
      clight=2.9979e5
      deg_per_rad=180./pi

C--   Convert the input parameters of the model into the proper units

      DO P=1,npar

C     Normalization constant for outer power law
      if (xib(P).le.xi1(P).or.xib(P).gt.xi2(P)) then
         pownorm=1.
      else
         pownorm=xib(P)**(q2(P)-q1(P))
      endif
      
C     From i(degrees) to sin i, cos i
      SINI = SIN(ANGI(P)/deg_per_rad)
      COSI = COS(ANGI(P)/deg_per_rad)

C     Broadening parameter from km/s to v/c
      BROAD(P)=BROAD(P)/clight   

C     From wind lambda(degrees) to lambda(radians), sin l, cos l, cot l
      ANGLAM(P)=ANGLAM(P)/deg_per_rad
      sinlam=sin(anglam(P))
      coslam=cos(anglam(P))
      cotlam=coslam/sinlam

C     Angles for spiral brightness patterm
      IF (AOBS(P).GE.0.) THEN
         XIREF=XI2(P)
         AOBS(P)=AOBS(P)/deg_per_rad
      ELSE
         XIREF=XI1(P)
         AOBS(P)=-AOBS(P)/deg_per_rad
      ENDIF
      TANPITCH=TAN(PITCH(P)/deg_per_rad)
      SIG=WIDTH(P)/SQRT(8.*LOG(2.))/deg_per_rad

C--   Construct the arrays of trigonometric functions. These functions
C--   depend only on the azimuth and will be very useful later in the
C--   CPU intensive loops.

      XIDEL = ALOG10(XI2(P)/XI1(P))/NSTEP/2. ! log steps in radius
      XIDEL = 10.**XIDEL               ! 1/2 radial step size
      XIDIFF = (XIDEL-1./XIDEL)
      PHISTEP = 2.*PI/NSTEP       ! phi step size

      DO I=1,NSTEP
         PHIP(I)=0.5*PHISTEP*(2*I-1)
         SINCOS(I)=SINI*COS(PHIP(I))
         SINSIN(I)=SINI*SIN(PHIP(I))
      ENDDO
      
C--> This is the heart of the program. Three nested loops compute the line
C    profile as a function of wavelength, by integrating over the surface of 
C    the disk.
      
      DO K=1,NPIX
         XX=(OLAMBDA(P)/WAVE(K))-1.
         CNU = 0.
         DO J=1,NSTEP
            XI=XI1(P)*XIDEL**(2*J-1)
            XISTEP=XI*XIDIFF
            ALPHA=SQRT(1.-(3./XI))
c            BETA=SQRT(1.-(2./XI))
            vinf=(1./sqrt(xi))/cappa
            radial=t0(P)*((xi/xi1(P))**eta(P))/(vinf/xi)
            if (xi.lt.xib(P)) then
               XIPOW=XI**(1-Q1(P))
            else
               XIPOW=POWNORM*XI**(1-Q2(P))
            endif
            sinlam=sin(anglam(P)*xi1(P)/xi)
            coslam=cos(anglam(P)*xi1(P)/xi)
            cotlam=coslam/sinlam
            PSI0=AOBS(P)+LOG10(XI/XIREF)/TANPITCH
            DO I=1,NSTEP
               ARMS=0.
       IF (AMP(P).NE.0.AND.XI.GE.XISPIN(P).AND.XI.LE.XISPOUT(P)) THEN
                  DO N=1,NARMS(P)
                     PSI=PSI0+2.*PI*FLOAT(N-1)/FLOAT(NARMS(P))
                     DPSI=ABS(PHIP(I)-PSI)
                     ARMS=ARMS+EXP(-DPSI*DPSI/2./SIG/SIG)
                     DPSI=2.*PI-ABS(PHIP(I)-PSI)
                     ARMS=ARMS+EXP(-DPSI*DPSI/2./SIG/SIG)
                  ENDDO
               ENDIF
               if (version.eq.'m') then !-> MC97 version
                  omega=sincos(i)*(sincos(i)+1.5*cappa*sinsin(i))
     &                 -cosi*(sincos(i)*cotlam+cosi
     &                        +0.5*cappa*sinsin(i)/sinlam)
               elseif (version.eq.'f') then !-> FEB12 version
                  omega=sincos(i)*(sincos(i)+1.5*cappa*sinsin(i))
     &                 +cosi*(sincos(i)/sinlam+cosi
     &                        +0.5*cappa*sinsin(i)/sinlam)
               else 
                  omega=1.
               endif
               tau=radial/abs(omega)
               direscprob=(1.-exp(-tau))/tau
               DOPPLER=1.+(SINSIN(I)/SQRT(XI))
               if (relativistic.eq.'y') then 
                  DOPPLER=ALPHA/DOPPLER ! relativistic
                  BOOST=DOPPLER*DOPPLER*DOPPLER
                  BEND=(1+((1-SINCOS(I))/(1+SINCOS(I)))/XI)
               else                  
                  DOPPLER=(1.-(SINSIN(I)/SQRT(XI)))/DOPPLER ! non-relativistic
                  DOPPLER=SQRT(DOPPLER) ! non-relativistic
                  BOOST=1.
                  BEND=1.
               endif
               EXPON=(1.+XX-DOPPLER)/DOPPLER/BROAD(P)
               EXPON=EXPON*EXPON/2.
               ARG=
     &              BOOST*             ! Doppler boosting
     &              XIPOW*             ! radial brightness profile
     &              direscprob*        ! wind opacity
     &              (1.+0.5*AMP(P)*ARMS)* ! spiral arm emissivity
     &              BEND*              ! light bending 
     &              EXP(-EXPON)        ! intrinsic line profile
               ELEMENT=ARG*XISTEP*PHISTEP
               CNU=CNU+ELEMENT
            ENDDO
         ENDDO
        
          flux(k,P)=cnu
         
c         write(*,*)  cnu
      ENDDO
      ENDDO
      return
      end

C END FILE PROFILEcirc2d.F
