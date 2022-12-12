import numpy as np
import profilemod
import profileell

def ellmodel_linefit(theta, x, y, yerr, lines, fixed, fitted, M, Cinv, lineprofs):
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profileell.ellprofile(params['maxstep'],params['npix'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['ell'],params['smooth'],params['phi0'],params['nstep'],params['olambda'],wave) 
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    return model

def model_linefit(theta, x, y, yerr, lines, fixed, fitted, M, Cinv, lineprofs):
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1'] 
    diskmodel = profilemod.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs'],params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],wave)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    return model

def model_linefit_broad(theta, x, y, yerr, lines, fixed, M, Cinv, lineprofs):
    A,w = theta
    broadmu = fixed
    broadmodel = A*np.max(y)*np.exp((x-broadmu)**2 * (-0.5/w**2)) 
    lhs = M@Cinv@(y-broadmodel)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:]):
        narrowmodel+=line*amp
    model = np.sum((broadmodel,narrowmodel),axis=0)
    return model

def log_likelihood_broad(theta, x, y, yerr, lines, fixed, M, Cinv, lineprofs): 
    A,w = theta
    broadmu = fixed
    broadmodel = A*np.max(y)*np.exp((x-broadmu)**2 * (-0.5/w**2)) 
    lhs = M@Cinv@(y-broadmodel)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:]):
        narrowmodel+=line*amp
    model = np.sum((broadmodel,narrowmodel),axis=0)
    sigma2 = yerr**2 
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def elllog_likelihood(theta, x, y, yerr, lines, fixed, fitted, M, Cinv, lineprofs):
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profileell.ellprofile(params['maxstep'],params['npix'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['ell'],params['smooth'],params['phi0']%360,params['nstep'],params['olambda'],wave)
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    sigma2 = yerr**2 #+ model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_likelihood(theta, x, y, yerr, lines, fixed, fitted, M, Cinv, lineprofs):
    fitted = dict(zip(fitted.keys(),theta)) 
    params = {**fitted, **fixed}
    xib0 = (params['xi2']*params['xi1']-params['xi1'])*params['xib']+params['xi1']
    diskmodel = profilemod.profile(params['maxstep'],params['xi1'],params['xi1']*params['xi2'],params['broad'],params['q1'],params['q2'],xib0,params['angi'],params['anglam'],params['t0'],params['eta'],params['version'],params['amp'],params['narms'],params['aobs'],params['pitch'],params['width'],params['xispin'],params['xispout'],params['nstep'],params['relativistic'],params['olambda'],params['npix'],wave)  
    M[-1] = diskmodel 
    lhs = M@Cinv@(y)
    rhs = M@Cinv@M.T
    amps = np.linalg.solve(rhs,lhs)
    narrowmodel = lineprofs[0] * amps[0]
    for line,amp in zip(lineprofs[1:],amps[1:-1]):
        narrowmodel+=line*amp
    model = np.sum((diskmodel*amps[-1],narrowmodel),axis=0)
    sigma2 = yerr**2 #+ model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

class log_prior(object):
    def __init__(self, mins, maxes):
        self.mins = mins
        self.maxes = maxes
    def __call__(self, theta):
        #for i,t in enumerate(theta):
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

class elllogprob(object):
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes, widths, lineprofs): 
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.M = np.empty((len(lines)+1,len(x)))
        for i in range(len(lines)):
            self.M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
        self.Cinv = np.eye(x.shape[0])*(1/yerr**2) #
        self.lineprofs = lineprofs
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        like = elllog_likelihood(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted, self.M, self.Cinv, self.lineprofs)   
        if like == np.nan:
            like = -np.inf
        return like+lp 
    def test(self,theta):
        modelout = ellmodel_linefit(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted, self.M, self.Cinv, self.lineprofs) 
        return modelout 



class logprob(object):
    def __init__(self, x, y, yerr, lines, fixed, fitted, mins, maxes, widths, lineprofs): 
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.fitted = fitted
        self.mins = mins
        self.maxes = maxes
        self.M = np.empty((len(lines)+1,len(x)))
        for i in range(len(lines)):
            self.M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
        self.Cinv = np.eye(x.shape[0])*(1/yerr**2) #
        self.lineprofs = lineprofs
        self.log_prior = log_prior(self.mins, self.maxes)
    def __call__(self,theta):
        lp = self.log_prior(theta)
        like = log_likelihood(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted, self.M, self.Cinv, self.lineprofs)  
        return like+lp 
    def test(self,theta):
        modelout = model_linefit(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.fitted, self.M, self.Cinv, self.lineprofs) 
        return modelout 

class logprob_broad(object):
    def __init__(self, x, y, yerr, lines, fixed, mins, maxes, widths, lineprofs): 
        self.x = x
        self.y = y
        self.yerr = yerr
        self.lines = lines
        self.fixed = fixed
        self.mins = mins
        self.maxes = maxes
        self.M = np.empty((len(lines),len(x)))
        for i in range(len(lines)):
            self.M[i] = np.exp(-0.5*((x-lines[i])/(widths[i]))**2)
        self.Cinv = np.eye(x.shape[0])*(1/yerr**2) #
        self.lineprofs = lineprofs
    def __call__(self,theta):
        like = log_likelihood_broad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.M, self.Cinv, self.lineprofs) 
        return like 
    def test(self,theta):
        modelout = model_linefit_broad(theta, self.x, self.y, self.yerr, self.lines, self.fixed, self.M, self.Cinv, self.lineprofs) 
        return modelout 


