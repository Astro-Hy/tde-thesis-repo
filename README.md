# Diskfit

A package to fit circular and elliptical accretion disk models to double-peaked broad line AGN spectra. It can also fit simple Gaussian broad line models.  

## Installation
Essential dependencies: numpy version>1.22.2, a fortran compiler such as gfortran, scipy, emcee, matplotlib. Optional dependencies: ptemcee, ultranest. 

To install: clone this repo, move to the top directory of the package, and run:

pip install .

Some notes on the installation:
	I am hoping the automated installation works correctly on different machines, but I'm very much in the testing phase here - any feedback very welcome!
	The pip install undergoes two installation steps. In the first step it installs the python module, and in the second step it compiles the fortran code. The pip install command may first give the warning 'ERROR: Failed building wheel for diskfit' before reinstalling successfully. I'm working on a fix for this but in the meantime, the best way to check that the install worked correctly is to run either the circular or elliptical disk fitting example jupyter notebooks in the 'examples' directory. 

If you encounter errors importing the diskmodels module, something has gone wrong with the f2py fortran compilation. If you have issues importing the diskfit module, something has gone wrong with the python installation. If you run into troubles I'm happy to help!

## Code usage
The best way to get started understanding how to use this code is the example jupyter notebooks in the examples directory - they will hopefully be easily modifiable for different use cases!

The log likelihood functions available in the diskfit module rely on the input of two dictionaries: one containing the names and values of 'fixed' parameters and one containing the names and starting values of 'fitted' parameters. All parameters must be defined in either one of these two dictionaries. The required parameters are as follows:

### For all models:
#### Parameters which can either be fitted or fixed:
'z': redshift

'narrowwidth': The width of the narrow emission lines in Angstroms

### For the Gaussian broad line model:
#### Parameters which can either be fitted or fixed:
'broadlam': The central rest wavelength of the Gaussian broadline

'broadwidth': The width of the broad emission line in Angstroms

### For the circular disk model:
#### Parameters which can either be fitted or fixed:
'olambda': nominal rest wavelength of the Halpha line (Angstrom)

'q1': inner emissivity powerlaw index

'q2': outer emissivity powerlaw index

'xib': power-law index break radius (XIB=0 causes Q=Q2 throughout)

'angi': disk inclination angle (degrees)

'xi1': inner disk radius (GM/c^2)

'xi2': outer disk radius (multiple of xi1)

'broad': broadening parameter (km/s)

Wind properties:

't0': optical depth normalization (0=no wind)

'eta': optical depth power-law index

'anglam': wind opening angle (degrees)

Spiral arms:

'amp': contrast of spiral arm (0=no arms)

'aobs': spiral orientation (degrees, +ve=outer, -ve=inner)

'pitch': pitch angle of spiral pattern (+ve=leading, -ve=trailing)

'width': angular width of arm (degrees)

'xispin': inner spiral arm radius radius (GM/c^2, 0=XI1)

'xispout': outer spiral arm radius radius (GM/c^2, 0=XI2)

#### Parameters which must be fixed:
'wavemin': minimum rest wavelength (Angstrom)

'wavemax': maximum rest wavelength (Angstrom)

'maxstep': maximum number of integration steps for diskmodel (int)

'nstep': integration steps (integer, <= 400)

'relativistic' = 'y' #include relativistic effects? (y/n) [y]

'normalization' = 'max' #profile normalization scheme (max/flux/none) [max]

'version' = 'f' #formula for escape probability for disk wind (f=Flohic, m=Murray)

'narms': number of arms (integer)

'npix' = integer number of wavelength points (the size of the wavelength array)

### For the elliptical disk model:
#### Parameters which can either be fitted or fixed:
'olambda': nominal rest wavelength of the Halpha line (Angstrom)

'q1': inner emissivity powerlaw index

'q2': outer emissivity powerlaw index

'xib': power-law index break radius (XIB=0 causes Q=Q2 throughout)

'angi': disk inclination angle (degrees)

'xi1': inner disk radius (GM/c^2)

'xi2': outer disk radius (multiple of xi1)

'broad': broadening parameter (km/s)

'ell': eccentricity (< 1), outer eccentricity if varying smoothly

'phi0': major axis orientation (0-360 deg)

#### Parameters which must be fixed:
'smooth' = 'y'#smoothly varying eccentricity (y/n)[n] 

'wavemin': minimum rest wavelength (Angstrom)

'wavemax': maximum rest wavelength (Angstrom)

'maxstep': maximum number of integration steps for diskmodel (int)

'nstep': integration steps (integer, <= 400)

'relativistic' = 'y' #include relativistic effects? (y/n) [y]

'normalization' = 'max' #profile normalization scheme (max/flux/none) [max]

'npix' = integer number of wavelength points (the size of the wavelength array)

Once we have assigned a value for each parameter above to a corresponding variable, we must then create dictionaries of fixed and fitted parameters.
```
fixed_labels = ['narrowwidth','q1','q2','xib','maxstep','anglam','t0','eta','version','amp','narms','aobs','pitch','width','xispin','xispout','nstep','relativistic','olambda','npix']
fixed_values = [narrowwidth,q1,q2,xib,maxstep,anglam,t0,eta,version,amp,narms,aobs,pitch,width,xispin,xispout,nstep,relativistic,olambda,npix]
fixed = dict(zip(fixed_labels,fixed_values))
fitted_labels = ['xi1','xi2','broad','angi','z']
initial = [xi1,xi2,broad,angi,z]
fitted = dict(zip(fitted_labels,initial))
```
The likelihood functions also require the observed wavelength array as an input. The observed wavelength is converted to the rest wavelength using the redshift parameter in the likelihood functions. The observed flux (in any units) and flux errors corresponding to the wavelength array are also required. Make sure to trim the wavelength and flux arrays so they only contain the region around H alpha.In the below example, the spectrum file contains 3 columns: observed wavelength (Angstroms), flux (after continuum subtraction with ppxf) and the flux errors.
```
from diskfit import utils
fn = 'data/ZTF18aahiqst_subtracted.txt'
wl,flux,fluxerr = utils.readspec(fn)
z = 0.0745 # Redshift
wavemin = 6300  # minimum rest wavelength (Angstrom)
wavemax = 6900  # maximum rest wavelength (Angstrom)
wave = wl/(1+z) # Convert the spectrum to rest frame wavelength
indwave = np.argwhere((wave>wavemin)&(wave<wavemax))[:,0]
wl = np.asarray(wl[indwave],dtype=np.float64)
flux = flux[indwave]
fluxerr = fluxerr[indwave]
npix = wl.shape[0]
``` 

### Setting uniform priors
The likelihood function currently supports a uniform prior for the fitted parameters. These are provided by two lists, one containing the minimum and one containing the maximum allowed values for each of the fitted parameters in order, e.g.
```
angimax = 89.9
angimin= 0.1
xi1min = 50
xi1max = 2000
xi2min = 1.01
xi2max = 40
broadmin = 0
broadmax = 1000
zmin = 0.99*z
zmax=1.01*z
diskmins = [xi1min,xi2min,broadmin,angimin,zmin]
diskmax = [xi1max,xi2max,broadmax,angimax,zmax]
```
 
### Solving for narrow line and diskmodel amplitudes
 
The log likelihood functions automatically solve the linear equation for the best fit amplitudes of the narrow lines and the broad line after calculating the disk model for the given parameters. The only requirement from the user is to specify the wavelength of the desired lines as follows:
```
NIIa = 6549.86
Halpha = 6564.614
NIIb = 6585.27
SIIa = 6718.29
SIIb = 6732.68
lines = [Halpha,NIIa,NIIb,SIIa,SIIb]
```
### Putting everything together

The log probability function can be initialized as follows for the circular, elliptical and Gaussian broad line models:
```
from diskfit import likelihood

lp = likelihood.logprob_circ(wl, flux, fluxerr, lines, fixed, fitted, mins, maxes)

lp = likelihood.logprob_ell(wl, flux, fluxerr, lines, fixed, fitted, mins, maxes)

lp = likelihood.logprob_broad(wl, flux, fluxerr, lines, fixed, fitted, mins, maxes)
```

where the inputs are:   
    
    wl: wavelengths (observed)

    flux: measured fluxes

    fluxerr: flux uncertainties

    lines: list of narrow emission line wavelengths to be included in the model

    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)

    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    mins: list of minimum values for the parameters in the fitted dictionary (in the same order).

    maxes: list of maximum values for the parameters in the fitted dictionary (in the same order).

It can then be called inside a fitting function with the input :
	theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)

For example, if we use emcee to fit the disk parameters we could write the following:

```
pos = initial + 1e-1 * initial * np.random.randn(20, initial.shape[0])
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lp, args=()
    )
```

where ``` initial``` is an array of values containing an initial guess for each fitted parameter.
