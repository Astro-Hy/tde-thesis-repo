from setuptools import find_packages
from numpy.distutils.core import setup as npsetup

install_requires = [
    "numpy>=1.22.2",
    "scipy",
    "setuptools<74"
]
python_requires="<3.12"

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('diskmodels', parent_package, top_path)
    config.add_extension('profileell', sources=['diskfit/profileell.pyf', 'diskfit/profileell.f'])
    config.add_extension('profileell2d', sources=['diskfit/profileell2d.pyf', 'diskfit/profileell2d.f'])
    config.add_extension('profilecirc', sources=['diskfit/profilecirc.pyf', 'diskfit/profilecirc.f'])
    config.add_extension('profilecirc2d', sources=['diskfit/profilecirc2d.pyf', 'diskfit/profilecirc2d.f'])
    return config

if __name__ == '__main__':
    # Generate configuration
    config = configuration(parent_package='', top_path=None).todict()

    # Add additional metadata
    config.update({
        "name": "diskfit",
        "packages": find_packages(),
        "include_package_data": True,
        "description": "Fitting spectra of double-peaked broad AGN emission lines with disk models",
        "author": "Charlotte Ward and Michael Eracleous",
        "author_email": "charlotte.ward42@gmail.com",
        "url": "https://github.com/charlotteaward/Diskfit",
        "keywords": ["astro", "AGN", "spectra", "disk"],
        "install_requires": install_requires,
        "python_requires": python_requires,
        "zip_safe": False,
    })

    # Call setup once
    npsetup(**config)