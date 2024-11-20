import numpy as np
from diskmodels import profileell,profilecirc,profileell2d,profilecirc2d
from diskfit import utils

def plot_linefit_circ(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.ones(len(lines))*params['narrowwidth']
    lineprofs = utils.build_line_profiles(x,lines,params['narrowwidth'])
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp 
    diskout = diskmodel*amps[-1]
    return diskout,narrowmodel

def plot_linefit_circ_fixeddoublet(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    M = np.empty((len(params['ratios'])+2,len(x)))
    M[0] = np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for p,i in zip(range(len(params['ratios'])),np.arange(1,len(lines),2)):        
        M[p+1] = np.sum((np.exp(-0.5*((x-lines[i])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[i+1])**2 * (-0.5/widths[i+1]**2) )),axis=0)
    M[0] = np.sum((M[0]/params['narrowfrac'],np.exp(-0.5*((x-lines[0])/(widths[len(lines)]))**2)),axis=0)
    for p,i in zip(range(len(params['ratios'])),np.arange(len(lines)+1,2*len(lines),2)):        
        M[p+1] = np.sum((M[p+1]/params['narrowfrac'],np.exp(-0.5*((x-lines[1+2*p])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[1+2*p])**2 * (-0.5/widths[i+1]**2) )),axis=0)

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:-1]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:-1]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*amps[-1]
    return diskout,narrowmodel

def plot_linefit_circ_fixeddoublet_freeamplitudes(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    '''
    M = np.empty((len(params['ratios'])+2,len(x)))
    M[0] = np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for p,i in zip(range(len(params['ratios'])),np.arange(1,len(lines),2)):        
        M[p+1] = np.sum((np.exp(-0.5*((x-lines[i])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[i+1])**2 * (-0.5/widths[i+1]**2) )),axis=0)
    M[0] = np.sum((M[0]/params['narrowfrac'],np.exp(-0.5*((x-lines[0])/(widths[len(lines)]))**2)),axis=0)
    for p,i in zip(range(len(params['ratios'])),np.arange(len(lines)+1,2*len(lines),2)):        
        M[p+1] = np.sum((M[p+1]/params['narrowfrac'],np.exp(-0.5*((x-lines[1+2*p])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[1+2*p])**2 * (-0.5/widths[i+1]**2) )),axis=0)

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    '''
    amps = params['amps_narrow']
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*params['amps_disk']
    return diskout,narrowmodel

def plot_linefit_circ_fixeddoublet_freeamplitudes2(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    amps = [params['Halphaflux'],params['NIIbflux'],params['SIIbflux']]
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*params['diskflux']
    return diskout,narrowmodel

def model_linefit_circ_fixeddoublet_freeamplitudes2(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    amps = [params['Halphaflux'],params['NIIbflux'],params['SIIbflux']]
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*params['diskflux']
    model = np.sum((diskout,narrowmodel),axis=0)
    return model


def model_linefit_circ(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.ones(len(lines))*params['narrowwidth']
    lineprofs = utils.build_line_profiles(x,lines,params['narrowwidth'])
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T 
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    return model

def model_linefit_circ_fixeddoublet(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    M = np.empty((len(params['ratios'])+2,len(x)))
    M[0] = np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for p,i in zip(range(len(params['ratios'])),np.arange(1,len(lines),2)):        
        M[p+1] = np.sum((np.exp(-0.5*((x-lines[i])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[i+1])**2 * (-0.5/widths[i+1]**2) )),axis=0)
    M[0] = np.sum((M[0]/params['narrowfrac'],np.exp(-0.5*((x-lines[0])/(widths[len(lines)]))**2)),axis=0)
    for p,i in zip(range(len(params['ratios'])),np.arange(len(lines)+1,2*len(lines),2)):        
        M[p+1] = np.sum((M[p+1]/params['narrowfrac'],np.exp(-0.5*((x-lines[1+2*p])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[1+2*p])**2 * (-0.5/widths[i+1]**2) )),axis=0)

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:-1]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:-1]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*amps[-1]
    '''
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.ones(len(lines))*params['narrowwidth']
    lineprofs = utils.build_doublet_profiles(x,lines,params['narrowwidth'],params['ratios'])
    M = np.empty((len(params['ratios'])+2,len(x)))
    M[0] = np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for p,i in zip(range(len(params['ratios'])),np.arange(1,len(widths),2)):
        M[p+1] = np.sum((np.exp(-0.5*((x-lines[i])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[i+1])**2 * (-0.5/widths[i+1]**2) )),axis=0)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp 
    diskout = diskmodel*amps[-1]
    '''
    model = np.sum((diskout,narrowmodel),axis=0)
    return model

def model_linefit_circ_fixeddoublet_amplitudes(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    M = np.empty((len(params['ratios'])+2,len(x)))
    M[0] = np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for p,i in zip(range(len(params['ratios'])),np.arange(1,len(lines),2)):        
        M[p+1] = np.sum((np.exp(-0.5*((x-lines[i])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[i+1])**2 * (-0.5/widths[i+1]**2) )),axis=0)
    M[0] = np.sum((M[0]/params['narrowfrac'],np.exp(-0.5*((x-lines[0])/(widths[len(lines)]))**2)),axis=0)
    for p,i in zip(range(len(params['ratios'])),np.arange(len(lines)+1,2*len(lines),2)):        
        M[p+1] = np.sum((M[p+1]/params['narrowfrac'],np.exp(-0.5*((x-lines[1+2*p])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[1+2*p])**2 * (-0.5/widths[i+1]**2) )),axis=0)

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    return amps[:-1],amps[-1]



def model_linefit_circ_fixeddoublet_freeamplitudes(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    amps = params['amps_narrow']
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*params['amps_disk']
    model = np.sum((diskout,narrowmodel),axis=0)
    return model



def plot_linefit_circ_broad(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-1)*params['narrowwidth'],np.ones(1)*params['broadwidth2'])) 
    lineprofs = utils.build_line_profiles(x,np.hstack((lines[:-1],lines[-1]+params['diff'])),widths) 
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)-1):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    M[-2] = np.exp(-0.5*((x-lines[-1]-params['diff'])/(widths[-1]))**2) 
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T 
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    diskout = diskmodel*amps[-1]
    broadout = lineprofs[-1]*amps[-2]
    narrowout = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:-1],amps[1:-2]):
        narrowout+=line*amp
    return diskout,broadout,narrowout

def plot_linefit_circ_freeratio(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs = utils.build_fixedratio_profiles(x,lines[:-2],widths[:-2],ratios)   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)   
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs * amps[0] 
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)  
    return np.sum((diskmodelb * amps[1],diskmodel*amps[-1]),axis=0),broadmodel,narrowmodel


def plot_linefit_circ_fixedratio(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    lineprofs = utils.build_fixedratio_profiles(x,lines[:-2],widths[:-2],params['ratios'])   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = params['ratios'][0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],params['ratios'][i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)   
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs * amps[0] 
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)  
    return np.sum((diskmodelb * amps[1],diskmodel*amps[-1]),axis=0),broadmodel,narrowmodel

def plot_linefit_circ_freeratio_twocompnarrow_nobroad(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x) 
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)],widths[:len(lines)],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],ratios)   
    M = np.empty((3,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    for i in range(0,len(lines)-3):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)
    M[1] = diskmodelb  
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0) 
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1]),axis=0),narrowmodel),axis=0) 
    return np.sum((diskmodelb * amps[1],diskmodel*amps[-1]),axis=0),narrowmodel




def plot_linefit_circ_freeratio_twocompnarrow(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi1b']*params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x) 
    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(len(lines)-2)*params['narrowwidth2'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)-2],widths[:len(lines)-2],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)-2],widths[len(lines)-2:2*len(lines)-2],ratios)   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-3):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)

    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0)
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0) 
    return np.sum((diskmodelb * amps[1],diskmodel*amps[-1]),axis=0),broadmodel,narrowmodel


def plot_linefit_circ_freeratio_twocompnarrow_twocompbroad(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi1b']*params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)

    widths = np.hstack((np.ones(len(lines)-4)*params['narrowwidth'],np.ones(len(lines)-4)*params['narrowwidth2'],np.ones(1)*params['broadwidth3b'],np.ones(1)*params['broadwidth3'],np.ones(1)*params['broadwidth2b'],np.ones(1)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)-4],widths[:len(lines)-4],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)-4],widths[len(lines)-4:2*len(lines)-4],ratios)   
    lines[-2] = lines[-2]+params['diffb']
    lines[-1] = lines[-1]+params['diff']
    lines[-4] = lines[-4]+params['diff2b']
    lines[-3] = lines[-3]+params['diff2']

    broadlineprofs = utils.build_line_profiles(x,lines[-4:-2],widths[-4:-2]) 
    broadlineprofs2 = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    '''
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-5):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-5):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)
    '''
    M[0] = diskmodelb 
    M[1] = np.sum((np.exp(-0.5*((x-lines[-3])/(widths[-3]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-4])/(widths[-4]))**2)),axis=0)  
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  
    amps0 = params['amps0']#0.11217721
    narrowmodel = np.sum((lineprofs1 * amps0 ,params['narrowfrac']*lineprofs2 * amps0),axis=0)

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(np.sum((y,-narrowmodel),axis=0))
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10)
    print('plot amplitudes',amps,params['narrowfrac'],widths,lines)

    broadmodel = np.sum((broadlineprofs[0] * amps[1]*params['broadfrac'],broadlineprofs[1]*amps[1],broadlineprofs2[0] * amps[2]*params['broadfrac'],broadlineprofs2[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[0],broadmodel),axis=0),narrowmodel),axis=0)
    '''
    M = np.empty((5,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-5):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-5):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)

    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-3])/(widths[-3]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-4])/(widths[-4]))**2)),axis=0)  
    M[3] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10)
    print(amps)
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0)
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2],broadlineprofs2[0] * amps[3]*params['broadfrac'],broadlineprofs2[1]*amps[3]),axis=0)   
    #model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0)
    '''
    return np.sum((diskmodelb * amps[0],diskmodel*amps[-1]),axis=0),broadmodel,narrowmodel


def model_linefit_circ_broad(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-1)*params['narrowwidth'],np.ones(1)*params['broadwidth2'])) 
    lineprofs = utils.build_line_profiles(x,np.hstack((lines[:-1],lines[-1]+params['diff'])),widths) 
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)-1):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    M[-2] = np.exp(-0.5*((x-lines[-1]-params['diff'])/(widths[-1]))**2) 
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    return model

def model_linefit_circ_freeratio_twocompnarrow_twocompbroad(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi1b']*params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-4)*params['narrowwidth'],np.ones(len(lines)-4)*params['narrowwidth2'],np.ones(1)*params['broadwidth3b'],np.ones(1)*params['broadwidth3'],np.ones(1)*params['broadwidth2b'],np.ones(1)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)-4],widths[:len(lines)-4],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)-4],widths[len(lines)-4:2*len(lines)-4],ratios)   
    lines[-2] = lines[-2]+params['diffb']
    lines[-1] = lines[-1]+params['diff']
    lines[-4] = lines[-4]+params['diff2b']
    lines[-3] = lines[-3]+params['diff2']

    broadlineprofs = utils.build_line_profiles(x,lines[-4:-2],widths[-4:-2]) 
    broadlineprofs2 = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 

    M = np.empty((4,len(x)))
    '''
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-5):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-5):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)
    '''
    M[0] = diskmodelb 
    M[1] = np.sum((np.exp(-0.5*((x-lines[-3])/(widths[-3]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-4])/(widths[-4]))**2)),axis=0)  
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  
    amps0 = params['amps0']#0.11217721
    narrowmodel = np.sum((lineprofs1 * amps0 ,params['narrowfrac']*lineprofs2 * amps0),axis=0)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(np.sum((y,-narrowmodel),axis=0))
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10)
    print('model',amps)
  
    broadmodel = np.sum((broadlineprofs[0] * amps[1]*params['broadfrac'],broadlineprofs[1]*amps[1],broadlineprofs2[0] * amps[2]*params['broadfrac'],broadlineprofs2[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[0],broadmodel),axis=0),narrowmodel),axis=0)
    return model



def model_linefit_circ_freeratio_twocompnarrow(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi1b']*params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)

    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(len(lines)-2)*params['narrowwidth2'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)-2],widths[:len(lines)-2],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)-2],widths[len(lines)-2:2*len(lines)-2],ratios)   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-3):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)

    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0)
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0)
    return model

def model_linefit_circ_freeratio_twocompnarrow_nobroad(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)],widths[:len(lines)],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],ratios)   
    M = np.empty((3,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    for i in range(0,len(lines)-3):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)
    M[1] = diskmodelb  
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0) 
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1]),axis=0),narrowmodel),axis=0) 
    return model


def model_linefit_circ_freeratio(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)

    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs = utils.build_fixedratio_profiles(x,lines[:-2],widths[:-2],ratios)   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs * amps[0] 
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0)
    return model


def model_linefit_circ_fixedratio(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)

    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    lineprofs = utils.build_fixedratio_profiles(x,lines[:-2],widths[:-2],params['ratios'])   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = params['ratios'][0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],params['ratios'][i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs * amps[0] 
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0)
    return model



def model_linefit_ell_broad(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the elliptical disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profileell.ellprofile(params['maxstep'],params['npix'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['ell'],params['smooth'],params['phi0'],params['nstep'],params['olambda'],x)  
    widths = np.hstack((np.ones(len(lines)-1)*params['narrowwidth'],np.ones(1)*params['broadwidth2'])) 
    lineprofs = utils.build_line_profiles(x,lines,widths)
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    return model



def model_linefit_ell(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the elliptical disk model given the parameters, find the best fit amplitudes for the disk and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profileell.ellprofile(params['maxstep'],params['npix'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['ell'],params['smooth'],params['phi0'],params['nstep'],params['olambda'],x) 
    widths = np.ones(len(lines))*params['narrowwidth']
    lineprofs = utils.build_line_profiles(x,lines,params['narrowwidth'])
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    return model


def model_linefit_broad(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and the width and central wavelength of a Gaussian broad line (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the Gaussian broad line model given the parameters, find the best fit amplitudes for the broad and the narrow lines, and return the full model as an array.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: array of model fluxes corresponding to the input wavelengths
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed} 
    x = w/(1+params['z'])
    broadmodel = np.exp((x-params['broadlam'])**2 * (-0.5/params['broadwidth']**2)) 
    widths = np.ones(len(lines))*params['narrowwidth']
    lineprofs = utils.build_line_profiles(x,lines,params['narrowwidth'])
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    #M[-2] = np.exp(-0.5*((x-lines[-1]+params['diff'])/(widths[-1]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = broadmodel
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T 
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10)  
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((broadmodel*amps[-1],narrowmodel),axis=0)
    return model

def loglikelihood_broad(theta, w, y, yerr, lines, fixed, fitted): 
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and the width and central wavelength of a Gaussian broad line (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the Gaussian broad line model given the parameters, solve for the best fit amplitudes for the broad and the narrow lines, and return the log likelihood of the data minus the model.
    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    x: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.

    Output:
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed} 
    x = w/(1+params['z'])
    broadmodel = np.exp((x-params['broadlam'])**2 * (-0.5/params['broadwidth']**2)) 
    widths = np.ones(len(lines))*params['narrowwidth']
    lineprofs = utils.build_line_profiles(x,lines,params['narrowwidth'])
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = broadmodel
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((broadmodel*amps[-1],narrowmodel),axis=0)
    sigma2 = yerr**2 
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def loglikelihood_ell(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the elliptical disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output:
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profileell.ellprofile(params['maxstep'],params['npix'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['ell'],params['smooth'],params['phi0']%360,params['nstep'],params['olambda'],x)
    widths = np.ones(len(lines))*params['narrowwidth']
    lineprofs = utils.build_line_profiles(x,lines,params['narrowwidth'])
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    sigma2 = yerr**2 
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def loglikelihood_ell_broad(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the elliptical disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output:
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profileell.ellprofile(params['maxstep'],params['npix'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['ell'],params['smooth'],params['phi0']%360,params['nstep'],params['olambda'],x)
    widths = np.hstack((np.ones(len(lines)-1)*params['narrowwidth'],np.ones(1)*params['broadwidth2'])) 
    lineprofs = utils.build_line_profiles(x,lines,widths)
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    sigma2 = yerr**2 
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def loglikelihood_circ_lq(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    #angi = (phases + 90) % (90) 
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)  
    widths = np.ones(len(lines))*params['narrowwidth']
    lineprofs = utils.build_line_profiles(x,lines,params['narrowwidth'])
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T 
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10)  
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    chi = (y-model)/yerr  
    return chi

def loglikelihood_circ_fixeddoublet_lq(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    M = np.empty((len(params['ratios'])+2,len(x)))
    M[0] = np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for p,i in zip(range(len(params['ratios'])),np.arange(1,len(lines),2)):        
        M[p+1] = np.sum((np.exp(-0.5*((x-lines[i])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[i+1])**2 * (-0.5/widths[i+1]**2) )),axis=0)
    M[0] = np.sum((M[0]/params['narrowfrac'],np.exp(-0.5*((x-lines[0])/(widths[len(lines)]))**2)),axis=0)
    for p,i in zip(range(len(params['ratios'])),np.arange(len(lines)+1,2*len(lines),2)):        
        M[p+1] = np.sum((M[p+1]/params['narrowfrac'],np.exp(-0.5*((x-lines[1+2*p])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[1+2*p])**2 * (-0.5/widths[i+1]**2) )),axis=0)

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:-1]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:-1]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*amps[-1]
    model = np.sum((diskout,narrowmodel),axis=0)
    chi = (y-model)/yerr  
    return chi


def loglikelihood_circ_fixeddoublet_freeamplitudes_lq(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    amps = params['amps_narrow']
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*params['amps_disk']
    model = np.sum((diskout,narrowmodel),axis=0)
    chi = (y-model)/yerr  
    return chi

def loglikelihood_circ_fixeddoublet_freeamplitudes2_lq(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    amps = [params['Halphaflux'],params['NIIbflux'],params['SIIbflux']]
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*params['diskflux']
    model = np.sum((diskout,narrowmodel),axis=0)
    chi = (y-model)/yerr  
    return chi


def loglikelihood_circ(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    #angi = (phases + 90) % (90) 
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)  
    widths = np.ones(len(lines))*params['narrowwidth']
    lineprofs = utils.build_line_profiles(x,lines,params['narrowwidth'])
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10)  
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    sigma2 = yerr**2  
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def loglikelihood_circ_fixeddoublet(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    M = np.empty((len(params['ratios'])+2,len(x)))
    M[0] = np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for p,i in zip(range(len(params['ratios'])),np.arange(1,len(lines),2)):        
        M[p+1] = np.sum((np.exp(-0.5*((x-lines[i])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[i+1])**2 * (-0.5/widths[i+1]**2) )),axis=0)
    M[0] = np.sum((M[0]/params['narrowfrac'],np.exp(-0.5*((x-lines[0])/(widths[len(lines)]))**2)),axis=0)
    for p,i in zip(range(len(params['ratios'])),np.arange(len(lines)+1,2*len(lines),2)):        
        M[p+1] = np.sum((M[p+1]/params['narrowfrac'],np.exp(-0.5*((x-lines[1+2*p])/(widths[i]))**2),params['ratios'][p]*np.exp( (x-lines[1+2*p])**2 * (-0.5/widths[i+1]**2) )),axis=0)

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:-1]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:-1]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*amps[-1]
    model = np.sum((diskout,narrowmodel),axis=0)
    sigma2 = yerr**2  
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def loglikelihood_circ_fixeddoublet_freeamplitudes(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    amps = params['amps_narrow']
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*params['amps_disk']
    model = np.sum((diskout,narrowmodel),axis=0)
    sigma2 = yerr**2  
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def loglikelihood_circ_fixeddoublet_freeamplitudes2(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lineprofs1 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[:len(lines)],params['ratios'])   
    lineprofs2 = utils.build_doublet_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],params['ratios'])    
    amps = [params['Halphaflux'],params['NIIbflux'],params['SIIbflux']]
    narrowmodel1 = lineprofs1[0] * amps[0]
    for line,amp in zip(lineprofs1[1:],amps[1:]):
        narrowmodel1+=line*amp 
    narrowmodel2 = lineprofs2[0] * amps[0]
    for line,amp in zip(lineprofs2[1:],amps[1:]):
        narrowmodel2+=line*amp 
    narrowmodel = np.sum((narrowmodel1,narrowmodel2),axis=0)
    diskout = diskmodel*params['diskflux']
    model = np.sum((diskout,narrowmodel),axis=0)
    sigma2 = yerr**2  
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))




def loglikelihood_circ_broad(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    params['t0'] = np.exp(params['t0'])
    x = w/(1+params['z'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)   
    widths = np.hstack((np.ones(len(lines)-1)*params['narrowwidth'],np.ones(1)*params['broadwidth2']))  
    lineprofs = utils.build_line_profiles(x,np.hstack((lines[:-1],lines[-1]+params['diff'])),widths)
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)-1):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    M[-2] = np.exp(-0.5*((x-lines[-1]-params['diff'])/(widths[-1]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    sigma2 = yerr**2  
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def loglikelihood_circ_broad_lq(theta, w, y, yerr, lines, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z']) 
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)   
    widths = np.hstack((np.ones(len(lines)-1)*params['narrowwidth'],np.ones(1)*params['broadwidth2']))  
    lineprofs = utils.build_line_profiles(x,np.hstack((lines[:-1],lines[-1]+params['diff'])),widths)
    M = np.empty((len(lines)+1,len(x)))
    for i in range(len(lines)-1):
        M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
    M[-2] = np.exp(-0.5*((x-lines[-1]-params['diff'])/(widths[-1]))**2)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T 
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    amps[1] = np.clip(amps[1], a_min=0.0, a_max=params['disklim'])
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0) 
    chi = (y-model)/yerr  
    return chi

def loglikelihood_circ_fixedratio(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z']) 
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)   
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    lineprofs = utils.build_fixedratio_profiles(x,lines[:-2],widths[:-2],params['ratios'])   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = params['ratios'][0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],params['ratios'][i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0) 
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs * amps[0] 
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0)
    sigma2 = yerr**2  
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def loglikelihood_circ_freeratio_lq(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z']) 
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)   
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs = utils.build_fixedratio_profiles(x,lines[:-2],widths[:-2],ratios)   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs * amps[0] 
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0)
    chi = (y-model)/yerr  
    return chi

def loglikelihood_circ_freeratio_twocompnarrow_lq(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z']) 
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)   
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi1b']*params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(len(lines)-2)*params['narrowwidth2'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)-2],widths[:len(lines)-2],params['ratios'])   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)-2],widths[len(lines)-2:2*len(lines)-2],params['ratios'])   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-3):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0)
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0) 
    chi = (y-model)/yerr  
    print(np.sum(chi**2))
    return chi

def loglikelihood_circ_freeratio_twocompnarrow_twocompbroad_lq(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """

    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi1b']*params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-4)*params['narrowwidth'],np.ones(len(lines)-4)*params['narrowwidth2'],np.ones(1)*params['broadwidth3b'],np.ones(1)*params['broadwidth3'],np.ones(1)*params['broadwidth2b'],np.ones(1)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)-4],widths[:len(lines)-4],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)-4],widths[len(lines)-4:2*len(lines)-4],ratios)   
    lines[-2] = lines[-2]+params['diffb']
    lines[-1] = lines[-1]+params['diff']
    lines[-4] = lines[-4]+params['diff2b']
    lines[-3] = lines[-3]+params['diff2']

    broadlineprofs = utils.build_line_profiles(x,lines[-4:-2],widths[-4:-2]) 
    broadlineprofs2 = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    
    M = np.empty((4,len(x)))
    '''
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-5):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-5):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)
    '''
    M[0] = diskmodelb 
    M[1] = np.sum((np.exp(-0.5*((x-lines[-3])/(widths[-3]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-4])/(widths[-4]))**2)),axis=0)  
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  
    amps0 = params['amps0']#0.11217721
    narrowmodel = np.sum((lineprofs1 * amps0 ,params['narrowfrac']*lineprofs2 * amps0),axis=0)

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(np.sum((y,-narrowmodel),axis=0))
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10)
     
    broadmodel = np.sum((broadlineprofs[0] * amps[1]*params['broadfrac'],broadlineprofs[1]*amps[1],broadlineprofs2[0] * amps[2]*params['broadfrac'],broadlineprofs2[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[0],broadmodel),axis=0),narrowmodel),axis=0)
    '''
    M = np.empty((5,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-5):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-5):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)

    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-3])/(widths[-3]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-4])/(widths[-4]))**2)),axis=0)  
    M[3] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0)
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2],broadlineprofs2[0] * amps[3]*params['broadfrac'],broadlineprofs2[1]*amps[3]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0) 
    '''
    chi = (y-model)/yerr  
    print(np.sum(chi**2),amps)
    return chi


def loglikelihood_circ_freeratio_twocompnarrow_nobroad_lq(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)],widths[:len(lines)],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],ratios)   
    M = np.empty((3,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    for i in range(0,len(lines)-3):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)
    M[1] = diskmodelb  
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0) 
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1]),axis=0),narrowmodel),axis=0) 
    chi = (y-model)/yerr  
    print(np.sum(chi**2))
    return chi




def loglikelihood_circ_freeratio_twocompnarrow_twocompbroad(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """

    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z'])
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi1b']*params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-4)*params['narrowwidth'],np.ones(len(lines)-4)*params['narrowwidth2'],np.ones(1)*params['broadwidth3b'],np.ones(1)*params['broadwidth3'],np.ones(1)*params['broadwidth2b'],np.ones(1)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)-4],widths[:len(lines)-4],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)-4],widths[len(lines)-4:2*len(lines)-4],ratios)   
    lines[-2] = lines[-2]+params['diffb']
    lines[-1] = lines[-1]+params['diff']
    lines[-4] = lines[-4]+params['diff2b']
    lines[-3] = lines[-3]+params['diff2']

    broadlineprofs = utils.build_line_profiles(x,lines[-4:-2],widths[-4:-2]) 
    broadlineprofs2 = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    '''
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-5):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-5):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)
    '''
    M[0] = diskmodelb 
    M[1] = np.sum((np.exp(-0.5*((x-lines[-3])/(widths[-3]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-4])/(widths[-4]))**2)),axis=0)  
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  
    amps0 = params['amps0']#0.11217721
    narrowmodel = np.sum((lineprofs1 * amps0 ,params['narrowfrac']*lineprofs2 * amps0),axis=0)

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(np.sum((y,-narrowmodel),axis=0))
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10)
     
    broadmodel = np.sum((broadlineprofs[0] * amps[1]*params['broadfrac'],broadlineprofs[1]*amps[1],broadlineprofs2[0] * amps[2]*params['broadfrac'],broadlineprofs2[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[0],broadmodel),axis=0),narrowmodel),axis=0)
    '''
    M = np.empty((5,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-5):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-5):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)

    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-3])/(widths[-3]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-4])/(widths[-4]))**2)),axis=0)  
    M[3] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  

    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0)
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2],broadlineprofs2[0] * amps[3]*params['broadfrac'],broadlineprofs2[1]*amps[3]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0) 
    '''
    sigma2 = yerr**2  
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))



def loglikelihood_circ_freeratio(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z']) 
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)   
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs = utils.build_fixedratio_profiles(x,lines[:-2],widths[:-2],ratios)   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs * amps[0] 
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0)
    sigma2 = yerr**2  
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def loglikelihood_circ_freeratio_twocompnarrow(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z']) 
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)   
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi1b']*params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(len(lines)-2)*params['narrowwidth2'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)-2],widths[:len(lines)-2],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)-2],widths[len(lines)-2:2*len(lines)-2],ratios)   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    #M[1] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[len(lines)-2]))**2)
    for i in range(0,len(lines)-3):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)  
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0)
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0) 
    sigma2 = yerr**2  
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def loglikelihood_circ_freeratio_twocompnarrow_nobroad(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z']) 
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)   
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1b'],params['xi2b'],params['broad'],params['q1b'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines))*params['narrowwidth'],np.ones(len(lines))*params['narrowwidth2'])) 
    lines = np.copy(linesin)
    ratios = np.asfarray([1.0,params['NIIb_Halpha']/params['NIIb_NIIa'],params['NIIb_Halpha'],params['SIIb_Halpha']/params['SIIb_SIIa'],params['SIIb_Halpha'],params['OIb_Halpha']/params['OIb_OIa'],params['OIb_Halpha'],params['OIIIb_Hbeta']*params['Halpha_Hbeta']/params['OIIIb_OIIIa'],params['OIIIb_Hbeta']*params['Halpha_Hbeta'],1/params['Halpha_Hbeta']]) 
    lineprofs1 = utils.build_fixedratio_profiles(x,lines[:len(lines)],widths[:len(lines)],ratios)   
    lineprofs2 = utils.build_fixedratio_profiles(x,lines[:len(lines)],widths[len(lines):2*len(lines)],ratios)   
    M = np.empty((3,len(x)))
    M[0] = ratios[0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    for i in range(0,len(lines)-3):
        M[0] = np.sum((M[0],params['narrowfrac']*ratios[i]*np.exp(-0.5*((x-lines[i])/(widths[i+len(lines)-2]))**2)),axis=0)
    M[1] = diskmodelb  
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = np.sum((lineprofs1 * amps[0] ,params['narrowfrac']*lineprofs2 * amps[0]),axis=0) 
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1]),axis=0),narrowmodel),axis=0) 
    sigma2 = yerr**2  
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))



def loglikelihood_circ_fixedratio(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z']) 
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)   
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    lineprofs = utils.build_fixedratio_profiles(x,lines[:-2],widths[:-2],params['ratios'])   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = params['ratios'][0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],params['ratios'][i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs * amps[0] 
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0)
    chi = (y-model)/yerr 
    print(np.sum(chi**2))
    return chi


def loglikelihood_circ_fixedratio_lq(theta, w, y, yerr, linesin, fixed, fitted):
    """
    Function which takes the wavelengths (w), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It will then calculate the circular disk model given the parameters, solve for the best fit amplitudes for the disk and the narrow lines, and return the log likelihood of the data minus the model.

    Inputs
    theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary)
    w: wavelengths (observed)
    y: measured fluxes
    yerr: flux uncertainties
    lines: list of narrow emission line wavelengths to be included in the model 
    fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
    fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
    
    Output
    model: float, the log likelihood of the data given the model.
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    x = w/(1+params['z']) 
    lines = np.copy(linesin)
    params['t0'] = np.exp(params['t0'])
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)   
    diskmodelb = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi']%180,params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs']%360,params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambdab'],params['npix'],x)
    widths = np.hstack((np.ones(len(lines)-2)*params['narrowwidth'],np.ones(2)*params['broadwidth2'])) 
    lines = np.copy(linesin)
    lineprofs = utils.build_fixedratio_profiles(x,lines[:-2],widths[:-2],params['ratios'])   
    lines[-2] = lines[-2]+params['diff']
    lines[-1] = lines[-1]+params['diff']
    broadlineprofs = utils.build_line_profiles(x,lines[-2:],widths[-2:]) 
    M = np.empty((4,len(x)))
    M[0] = params['ratios'][0]*np.exp(-0.5*((x-lines[0])/(widths[0]))**2)
    for i in range(1,len(lines)-3):
        M[0] = np.sum((M[0],params['ratios'][i]*np.exp(-0.5*((x-lines[i])/(widths[i]))**2)),axis=0)
    M[1] = diskmodelb 
    M[2] = np.sum((np.exp(-0.5*((x-lines[-1])/(widths[-1]))**2 ),params['broadfrac']*np.exp(-0.5*((x-lines[-2])/(widths[-2]))**2)),axis=0)
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.clip(np.linalg.solve(rhs,lhs),a_min=0.0, a_max=1e10) 
    narrowmodel = lineprofs * amps[0] 
    broadmodel = np.sum((broadlineprofs[0] * amps[2]*params['broadfrac'],broadlineprofs[1]*amps[2]),axis=0)   
    model = np.sum((np.sum((diskmodel*amps[-1],diskmodelb*amps[1],broadmodel),axis=0),narrowmodel),axis=0)
    chi = (y-model)/yerr 
    print(np.sum(chi**2))
    return chi




class log_prior(object):
    '''
    Function to establish a uniform prior between minimum and maximum parameter values for each fitted parameter. The initialization function takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary. When called, the function returns a prior of 0 if all parameters are within the bounsd, and a prior of -inf if any parameters are out of the bounds. 
    '''
    def __init__(self, mins, maxes):
        '''
        Input:
        mins: list of minimum values for the parameters in the fitted dictionary (in the same order).
        maxes: list of maximum values for the parameters in the fitted dictionary (in the same order).
        '''
        self.mins = mins
        self.maxes = maxes
    def __call__(self, theta): 
        '''
        Input:
        theta: Array of values of fitted parameters (in the same order as the fitted dictionary).
        Output:
        float corresponding to the uniform prior
        '''
        if np.any(theta<self.mins) or np.any(theta>self.maxes): 
            return -np.inf
        return 0.0        

class logprob_ell_broad(object):
    '''
    A class to return the log probability of a elliptical disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        like = loglikelihood_ell_broad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)#), self.M, self.Cinv, self.lineprofs)   
        if like == np.nan:
            like = -np.inf
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_ell_broad(theta, self.x, self.y, self.yerr, self.lines,self.fixed, self.fitted) 
        return modelout 


class logprob_ell(object):
    '''
    A class to return the log probability of a elliptical disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        like = loglikelihood_ell(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)#), self.M, self.Cinv, self.lineprofs)   
        if like == np.nan:
            like = -np.inf
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_ell(theta, self.x, self.y, self.yerr, self.lines,self.fixed, self.fitted) 
        return modelout 

class logprob_circ_lq(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_lq(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10+lp)
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,narrowout = plot_linefit_circ(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,narrowout

class logprob_circ_fixeddoublet_lq(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_fixeddoublet_lq(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10+lp)
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_fixeddoublet(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,narrowout = plot_linefit_circ_fixeddoublet(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 

        return diskout,narrowout




class logprob_circ(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.isnan(like+lp):
            return -np.inf
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 


class logprob_circ_fixeddoublet(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_fixeddoublet(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.isnan(like+lp):
            return -np.inf
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_fixeddoublet(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def amplitudes(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        ampsn,ampsd = model_linefit_circ_fixeddoublet_amplitudes(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return ampsn,ampsd 





class logprob_circ_broad(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_broad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.isnan(like+lp):
            return -np.inf
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_broad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_broad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout

class logprob_circ_fixedratio(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_fixedratio(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10)+lp
        #    return -np.inf  
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_fixedratio(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_fixedratio(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout

class logprob_circ_freeratio_twocompnarrow_nobroad(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_freeratio_twocompnarrow_nobroad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10+lp)
        #    return -np.inf  
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_freeratio_twocompnarrow_nobroad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_freeratio_twocompnarrow_nobroad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout
 
class logprob_circ_fixeddoublet_freeamplitudes(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_fixeddoublet_freeamplitudes(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            #return np.full(len(like),1e10+lp)
            return -np.inf  
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_fixeddoublet_freeamplitudes(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_fixeddoublet_freeamplitudes(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout
 
class logprob_circ_fixeddoublet_freeamplitudes2(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_fixeddoublet_freeamplitudes2(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            #return np.full(len(like),1e10+lp)
            return -np.inf  
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_fixeddoublet_freeamplitudes2(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,narrowout = plot_linefit_circ_fixeddoublet_freeamplitudes2(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,narrowout
 
class logprob_circ_freeratio_twocompnarrow(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_freeratio_twocompnarrow(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            #return np.full(len(like),1e10+lp)
            return -np.inf  
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_freeratio_twocompnarrow(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_freeratio_twocompnarrow(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout
 


class logprob_circ_freeratio(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_freeratio(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10+lp)
        #    return -np.inf  
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_freeratio(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_freeratio(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout
 
class logprob_circ_freeratio_twocompnarrow_nobroad_lq(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_freeratio_twocompnarrow_nobroad_lq(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10)#+lp):
        #    return -np.inf  
        return like#+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_freeratio_twocompnarrow_nobroad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,narrowout = plot_linefit_circ_freeratio_twocompnarrow_nobroad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,narrowout
 



class logprob_circ_freeratio_twocompnarrow_lq(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_freeratio_twocompnarrow_lq(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10)#+lp):
        #    return -np.inf  
        return like#+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_freeratio_twocompnarrow(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_freeratio_twocompnarrow(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout

class logprob_circ_freeratio_twocompnarrow_twocompbroad(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_freeratio_twocompnarrow_twocompbroad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            #return np.full(len(like),1e10)#+lp):
            return -np.inf  
        return like+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_freeratio_twocompnarrow_twocompbroad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_freeratio_twocompnarrow_twocompbroad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout
 
class logprob_circ_freeratio_twocompnarrow_twocompbroad_lq(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_freeratio_twocompnarrow_twocompbroad_lq(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10)#+lp):
        #    return -np.inf  
        return like#+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_freeratio_twocompnarrow_twocompbroad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_freeratio_twocompnarrow_twocompbroad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout
 

class logprob_circ_freeratio_lq(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_freeratio_lq(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10)#+lp):
        #    return -np.inf  
        return like#+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_freeratio(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_freeratio(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout
 

class logprob_circ_fixedratio_lq(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_fixedratio_lq(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10)#+lp):
        #    return -np.inf  
        return like#+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_fixedratio(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_fixedratio(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout
 



class logprob_circ_broad_lq(object):
    '''
    A class to return the log probability of a circular disk model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of disk parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''    
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_circ_broad_lq(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)
        if np.any(np.isnan(like)):
            return np.full(len(like),1e10)#+lp):
        #    return -np.inf 
        return like#+lp 
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_circ_broad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 
    def plot(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        diskout,broadout,narrowout = plot_linefit_circ_broad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return diskout,broadout,narrowout
 




class logprob_broad(object):
    '''
    A class to return the log probability of a Gaussian broad line model using the corresponding log likelihood and log prior functions. The initialization function takes the observed wavelengths (x), fluxes (y), flux errors (yerr) of a spectrum, and a set of Gaussian broad line parameters (as well as redshift and narrow emission line width) distributed amongst two dictionaries (fitted and fixed). It also takes a list of minimum values, and a list of maximum values, which should correspond to the parameters listed in the 'fitted' dictionary.
    '''
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes): 
        '''
        Inputs 
            x: wavelengths (observed)
            y: measured fluxes
            yerr: flux uncertainties
            lines: list of narrow emission line wavelengths to be included in the model 
            fixed: dictionary of fixed disk parameters (parameter labels: parameter values)
            fitted: dictionary of fitted disk parameters (parameter labels: parameter values) The values in this dictionary will be updated to the array values carried in theta.
        '''  
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        '''
        Input
        theta: np.array containing updated fitted broad line parameters (corresponding to the labels in the fitted dictionary).
        
        Output
        float containing sum of the log prior and the log likelihood of the data given the model.
        '''
        lp = self.log_prior(theta)
        like = loglikelihood_broad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted)#, self.M, self.Cinv, self.lineprofs) 
        if like + lp == np.nan:
            return -np.inf
        return like+lp  
    def test(self,theta):
        '''
        For plotting of models
        Input
        theta: np.array containing updated fitted disk parameters (corresponding to the labels in the fitted dictionary).
        Output:
        Array containing the model fluxes corresponding to the given parameters
        '''
        modelout = model_linefit_broad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted) 
        return modelout 


