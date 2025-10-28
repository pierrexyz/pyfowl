from setuptools import setup

setup(
    name='pyfowl',
    version='0.1.0',
    description='Python code for Fast Observables in Weak Lensing',
    author="Pierre Zhang and Guido D'Amico",
    license='MIT',
    packages=['pyfowl'],
    install_requires=['numpy', 'scipy', 'pyyaml', 'astropy', 'mpmath'],
    package_dir = {'pyfowl': 'pyfowl'},
    zip_safe=False,

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Programming Language :: Python",
    ],
)
