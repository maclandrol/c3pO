from setuptools import setup
from setuptools import find_packages

setup(
    name="c3pO",
    packages=find_packages(),
    python_requires=">=3.6",
    include_package_data=True,
    version="0.0.1",
    description="Python implementation of the conformal prediction framework.",
    author="Emmanuel Noutahi",
    author_email="henrik.linusson@gmail.com",
    url="https://github.com/maclandrol/c3pO",
    install_requires=["numpy", "scikit-learn", "scipy", "pandas"],
    keywords=["conformal prediction", "machine learning", "classification", "regression"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
)

# Authors: Henrik Linusson
