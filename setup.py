import pathlib 
from setuptools import setup, find_packages 

HERE = pathlib.Path(__file__).parent 
VERSION = '0.2.1' 
PACKAGE_NAME = 'multi_dimensional_clustering' 
AUTHOR = 'Pieter Meyer' 
AUTHOR_EMAIL = 'you@email.com' 
URL = 'https://github.com/PieterMey/multi_dimensional_clustering' 
LICENSE = 'Apache License 2.0' 
DESCRIPTION = 'Multi-dimensional clustering visualization tool.' 
LONG_DESCRIPTION = (HERE / "README.md").read_text() 
LONG_DESC_TYPE = "text/markdown" 
INSTALL_REQUIRES = [ 
      'numpy', 
      'matplotlib',
      'plotly',
      'scikit-learn',
      'seaborn' 
] 
setup(name=PACKAGE_NAME, 
      version=VERSION, 
      description=DESCRIPTION, 
      long_description=LONG_DESCRIPTION, 
      long_description_content_type=LONG_DESC_TYPE, 
      author=AUTHOR, 
      license=LICENSE, 
      author_email=AUTHOR_EMAIL, 
      url=URL, 
      install_requires=INSTALL_REQUIRES, 
      packages=find_packages() 
      )
