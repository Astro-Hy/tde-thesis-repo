import os
from setuptools import setup, Extension, find_packages
import sys
import setuptools
#import version
import numpy.f2py
from numpy.distutils.core import setup as npsetup
install_requires = ["numpy", "scipy", "ptemcee", "emcee", "diskmodels", 'ultranest']

#r = numpy.f2py.run_main(['-c','diskfit/profileell.pyf','-m','diskfit/profileell.so','diskfit/profileell.f'])

def configuration(parent_package='', top_path=None, package_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('diskmodels',parent_package, top_path)
    config.add_extension('profileell', sources=['diskfit/profileell.pyf','diskfit/profileell.f'])
    config.add_extension('profileell2d', sources=['diskfit/profileell2d.pyf','diskfit/profileell2d.f'])
    config.add_extension('profilecirc', sources=['diskfit/profilecirc.pyf','diskfit/profilecirc.f'])
    config.add_extension('profilecirc2d', sources=['diskfit/profilecirc2d.pyf','diskfit/profilecirc2d.f'])
    return config

if __name__ == '__main__':
    
    npsetup(**configuration(parent_package='',top_path='diskfit/',package_path='diskfit/').todict())
    packages = find_packages() 
    npsetup(
        name="diskfit",
        packages=packages,
        include_package_data=True,
        description="Fitting spectra of double-peaked broad AGN emission lines with disk models",
        author="Charlotte Ward and Michael Eracleous",
        author_email="charlotte.ward42@gmail.com",
        url="https://github.com/charlotteaward/Diskfit",
        keywords=["astro", "AGN", "spectra", "disk"],
        ext_package=['diskmodels'],
        #ext_modules=,
        install_requires=install_requires,
        #cmdclass={"build_ext": BuildExt},
        zip_safe=False,
        #version=pkg_version,
    )
    
