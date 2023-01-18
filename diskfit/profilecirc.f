C FILE: PROFILEcirc.F
      SUBROUTINE PROFILE(maxstep, 
     &     xi1,xi2,broad,q1,q2,xib,angi, 
     &     anglam,t0,eta,version,       
     &     amp,narms,aobs,pitch,width,xispin,xispout,
     &     nstep,relativistic,olambda,              
     &     npix,wave,flux)
C
C     CALCULATE FIRST N FIBONACCI NUMBERS
C
      INTEGER maxstep,narms, nstep, npix
      REAL*8 SINCOS(MAXSTEP),SINSIN(MAXSTEP)
      REAL*8 CNU,DOPPLER,BOOST,BEND,XX,ELEMENT
      real*8 sini,cosi,coti,tani,sinlam,cotlam
      real*8 omega,tau,direscprob,arg,radial
      real*8 EXPON
      real*4 xi1,xi2,broad,q1,q2,xib
      real*4 xi
      real*4 angi,anglam,t0,eta,amp,aobs
      real*4 pitch,width,xispin,xispout,olambda
      REAL*4 PHIP(MAXSTEP),WAVE(npix)
      REAL*8 flux(npix)
      character*1 version,relativistic

Cf2py intent(in) maxstep
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
Cf2py intent(in) npix
Cf2py depend(PHIP) maxstep
Cf2py intent(out) flux
Cf2py intent(in) wave
Cf2py depend(flux) npix
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

C     Normalization constant for outer power law
      if (xib.le.xi1.or.xib.gt.xi2) then
         pownorm=1.
      else
         pownorm=xib**(q2-q1)
      endif
      
C     From i(degrees) to sin i, cos i
      SINI = SIN(ANGI/deg_per_rad)
      COSI = COS(ANGI/deg_per_rad)

C     Broadening parameter from km/s to v/c
      BROAD=BROAD/clight   

C     From wind lambda(degrees) to lambda(radians), sin l, cos l, cot l
      ANGLAM=ANGLAM/deg_per_rad
      sinlam=sin(anglam)
      coslam=cos(anglam)
      cotlam=coslam/sinlam

C     Angles for spiral brightness patterm
      IF (AOBS.GE.0.) THEN
         XIREF=XI2
         AOBS=AOBS/deg_per_rad
      ELSE
         XIREF=XI1
         AOBS=-AOBS/deg_per_rad
      ENDIF
      TANPITCH=TAN(PITCH/deg_per_rad)
      SIG=WIDTH/SQRT(8.*LOG(2.))/deg_per_rad

C--   Construct the arrays of trigonometric functions. These functions
C--   depend only on the azimuth and will be very useful later in the
C--   CPU intensive loops.

      XIDEL = ALOG10(XI2/XI1)/NSTEP/2. ! log steps in radius
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
         XX=(OLAMBDA/WAVE(K))-1.
         CNU = 0.
         DO J=1,NSTEP
            XI=XI1*XIDEL**(2*J-1)
            XISTEP=XI*XIDIFF
            ALPHA=SQRT(1.-(3./XI))
c            BETA=SQRT(1.-(2./XI))
            vinf=(1./sqrt(xi))/cappa
            radial=t0*((xi/xi1)**eta)/(vinf/xi)
            if (xi.lt.xib) then
               XIPOW=XI**(1-Q1)
            else
               XIPOW=POWNORM*XI**(1-Q2)
            endif
            sinlam=sin(anglam*xi1/xi)
            coslam=cos(anglam*xi1/xi)
            cotlam=coslam/sinlam
            PSI0=AOBS+LOG10(XI/XIREF)/TANPITCH
            DO I=1,NSTEP
               ARMS=0.
               IF (AMP.NE.0.AND.XI.GE.XISPIN.AND.XI.LE.XISPOUT) THEN
                  DO N=1,NARMS
                     PSI=PSI0+2.*PI*FLOAT(N-1)/FLOAT(NARMS)
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
               EXPON=(1.+XX-DOPPLER)/DOPPLER/BROAD
               EXPON=EXPON*EXPON/2.
               ARG=
     &              BOOST*             ! Doppler boosting
     &              XIPOW*             ! radial brightness profile
     &              direscprob*        ! wind opacity
     &              (1.+0.5*AMP*ARMS)* ! spiral arm emissivity
     &              BEND*              ! light bending 
     &              EXP(-EXPON)        ! intrinsic line profile
               ELEMENT=ARG*XISTEP*PHISTEP
               CNU=CNU+ELEMENT
            ENDDO
         ENDDO
         flux(k)=cnu
c         write(*,*)  cnu
      ENDDO

      return
      end

      
C======================================================================
      subroutine setflux
     &     (fluxunits,normalization,pcnu,
     &     npix,wave,flux)
C======================================================================
C Put the flux density in the units requested by the user and normalize
C the line profile. The flux units are specified by the parameter
C FLUXUNITS (fnu/flam) and the normalization is specified by
C NORMALIZATION (max/flux/none). The array PCNU carries the model
C profile computed in the main program (in units of f-nu). The array
C FLUX carries the final profile after unit conversion and
C normalization.
C----------------------------------------------------------------------

      character*5 normalization,fluxunits
      real wave(*),flux(*),pcnu(*)

      clightarg=2.9979e-8       ! speed of light in 1e5 km/s

C-- Put the flux density in the units requested by the user. The PCNU
C-- array carries the computed profile in f-nu. The FLUX array will
C-- carry the final profile in the units requested by the user. The
C-- default is f-nu.

      if (fluxunits.eq.'flam') then
         do i=1,npix
            flux(i)=clightarg*pcnu(i)/(wave(i)*wave(i))
         enddo
      else
         do i=1,npix
            flux(i)=pcnu(i)
         enddo
      endif

C-- Normalize the line profile. If NORMALIZATION=max, find the maximum
C-- of the FLUX array (which is already in the units requested by the
C-- user) and divide FLUX array by that. If NORMALIZATION=flux, find the
C-- integrated flux of the PCNU array (which is always in f-nu) and
C-- divide the FLUX array by that. Otherwise do nothing so that the
C-- profile keeps its original normalization (the last option is the
C-- default).
      
      if (normalization.eq.'max') then
         fmax=flux(1)
         do i=1,npix
            fmax=max(fmax,flux(i))
         enddo
         do i=1,npix
            flux(i)=flux(i)/fmax
         enddo
      elseif (normalization.eq.'flux') then
         ftot=0.
         do i=1,npix
            if (i.eq.1) then
               dw=wave(2)-wave(1)
            elseif (i.eq.npix) then
               dw=wave(npix)-wave(npix-1)
            else
               dw=0.5*(wave(i+1)-wave(i-1))
            endif
            ftot=ftot+clightarg*dw*pcnu(i)/(wave(i)*wave(i))
         enddo
         do i=1,npix
            flux(i)=flux(i)/ftot
         enddo
      endif

      END
C END FILE PROFILECIRC.F
