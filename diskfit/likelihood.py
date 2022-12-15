import numpy as np
from diskmodels import profileell,profilecirc,profileell2d,profilecirc2d
from diskfit import utils
def model_linefit_ell(theta, x, y, yerr, lines, narrowmu, fixed, fitted):
    """
    Function to take find the best fit amplitudes for disk + narrow line emission for a given set of disk parameters and narrow line wavelengths and a given spectrum, and return the full model
    theta: np.array containing fitted disk parameters
    x: wavelengths
    y: measured fluxes
    yerr: flux uncertainties
    fixed: dictionary of fixed disk parameters
    M: 
    Cinv: Inverse covariance matrix
    lineprofs: 
    """
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profileell.ellprofile(params['maxstep'],params['npix'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['ell'],params['smooth'],params['phi0'],params['nstep'],params['olambda'],x) 
    widths = np.ones(len(lines))*narrowmu
    lineprofs = utils.build_line_profiles(x,lines,narrowmu)
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

def model_linefit_circ(theta, x, y, yerr, lines, narrowmu, fixed, fitted):
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs'],params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)
    widths = np.ones(len(lines))*narrowmu
    lineprofs = utils.build_line_profiles(x,lines,narrowmu)
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

def model_linefit_broad(theta, x, y, yerr, lines, narrowmu, fixed, fitted):
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed} 
    broadmodel = np.exp((x-params['broadmu'])**2 * (-0.5/params['width']**2)) 
    widths = np.ones(len(lines))*narrowmu
    lineprofs = utils.build_line_profiles(x,lines,narrowmu)
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

def loglikelihood_broad(theta, x, y, yerr, lines, narrowmu,fixed, fitted, M, Cinv, lineprofs): 
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed} 
    broadmodel = np.exp((x-params['broadmu'])**2 * (-0.5/params['width']**2)) 
    widths = np.ones(len(lines))*narrowmu
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

def loglikelihood_ell(theta, x, y, yerr, lines, fixed, fitted, M, Cinv, lineprofs):
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profileell.ellprofile(params['maxstep'],params['npix'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['ell'],params['smooth'],params['phi0']%360,params['nstep'],params['olambda'],x)
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

def loglikelihood_circ(theta, x, y, yerr, lines, fixed, fitted, M, Cinv, lineprofs):
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilecirc.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs'],params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],x)  
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
    def __init__(self, mins, maxes):
        self.mins = mins
        self.maxes = maxes
    def __call__(self, theta): 
        if np.any(theta<self.mins) or np.any(theta>self.maxes): 
            return -np.inf
        return 0.0        

def log_probability(theta, x, y, yerr, lines, fixed):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    like = log_likelihood(theta, x, y, yerr, lines, fixed, fitted) 
    if like == np.nan:
        like = -np.inf
    return lp + like 

class logprob_ell(object):
    def __init__(self, x, y, yerr, lines, narrowmu, fixed, fitted, mins, maxes): 
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.widths = np.ones(len(lines))*narrowmu
        self.narrowmu = narrowmu
        self.lineprofs = utils.build_line_profiles(x,lines,narrowmu)
        self.M = np.empty((len(lines)+1,len(x)))
        for i in range(len(lines)):
            self.M[i] = np.exp(-0.5*((x-lines[i])/(self.widths[i]))**2)
        self.Cinv = np.eye(x.shape[0])*(1/yerr**2) # 
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        like = loglikelihood_ell(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted, self.M, self.Cinv, self.lineprofs)   
        if like == np.nan:
            like = -np.inf
        return like+lp 
    def test(self,theta):
        modelout = model_linefit_ell(theta, self.x, self.y, self.yerr, self.lines, self.narrowmu ,self.fixed, self.fitted) 
        return modelout 

class logprob_circ(object):
    def __init__(self, x, y, yerr, lines, narrowmu, fixed, fitted, mins, maxes): 
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.widths = np.ones(len(lines))*narrowmu
        self.narrowmu = narrowmu
        self.lineprofs = utils.build_line_profiles(x,lines,narrowmu)
        self.M = np.empty((len(lines)+1,len(x)))
        for i in range(len(lines)):
            self.M[i] = np.exp(-0.5*((x-lines[i])/(self.widths[i]))**2)
        self.Cinv = np.eye(x.shape[0])*(1/yerr**2) # 
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        lp = self.log_prior(theta)
        like = loglikelihood_circ(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted, self.M, self.Cinv, self.lineprofs)  
        return like+lp 
    def test(self,theta):
        modelout = model_linefit_circ(theta, self.x, self.y, self.yerr, self.lines, self.narrowmu, self.fixed, self.fitted) 
        return modelout 

class logprob_broad(object):
    def __init__(self, x, y, yerr, lines, narrowmu, fixed, fitted, mins, maxes): 
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.widths = np.ones(len(lines))*narrowmu
        self.lineprofs = utils.build_line_profiles(x,lines,narrowmu)
        self.M = np.empty((len(lines),len(x)))
        for i in range(len(lines)):
            self.M[i] = np.exp(-0.5*((x-lines[i])/(self.widths[i]))**2)
        self.Cinv = np.eye(x.shape[0])*(1/yerr**2) 
    def __call__(self,theta):
        like = loglikelihood_broad(theta, self.x, self.y, self.yerr, self.lines, self.narrowmu, self.fixed, self.fitted, self.M, self.Cinv, self.lineprofs) 
        return like 
    def test(self,theta):
        modelout = model_linefit_broad(theta, self.x, self.y, self.yerr, self.lines, self.narrowmu, self.fixed, self.fitted) 
        return modelout 


