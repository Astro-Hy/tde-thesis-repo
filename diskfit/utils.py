import numpy as np

def readspec(fn):
    data = np.loadtxt(fn)
    return data[:,0],data[:,1],data[:,2]

def gaussian(x, mu, sig, amp):
    return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def build_line_profiles(x, mus, widths):
    widths = np.atleast_1d(widths)
    if len(widths) == 1:
        widths = np.full(len(mus),widths)
    profiles = np.empty((len(mus),len(x)))
    for i,(m,w) in enumerate(zip(mus,widths)):
        profiles[i] = np.exp( (x-m)**2 * (-0.5/w**2) )
    return profiles

def get_narrowline_matrices(wave,lines,narrowmu,fluxerr):
    widths = np.ones(len(lines))*narrowmu
    lineprof = build_line_profiles(wave,lines,narrowmu)
    M = np.empty((len(lines)+1,len(wave)))
    for i in range(len(lines)):
        M[i] = np.exp(-0.5*((wave-lines[i])/(widths[i]))**2)
    Cinv = np.eye(wave.shape[0])*(1/fluxerr**2)
    return lineprof,M,Cinv
