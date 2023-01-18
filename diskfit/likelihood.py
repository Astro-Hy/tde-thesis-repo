import numpy as np
from diskmodels import profileell,profilecirc,profileell2d,profilecirc2d
from diskfit import utils

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
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs'],params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
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
    diskmodel = profileell.ellprofile(params['maxstep'],params['npix'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['ell'],params['smooth'],params['phi0'],params['nstep'],params['olambda'],x) 
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
    Cinv = np.eye(x.shape[0])*(1/yerr**2)
    M[-1] = broadmodel
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
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
    diskmodel = profileell.ellprofile(params['maxstep'],params['npix'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['ell'],params['smooth'],params['phi0']%360,params['nstep'],params['olambda'],x)
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
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs'],params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)  
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


