import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

setup(name='cgoptimizer',
      version='1.0.0',
      description='Critical Gradient Optimizers',
      url='https://github.com/chandar-lab/CGOptimizer',
      author='Paul-Aymeric McRae, Prasanna Parthasarathi',
      author_email='paul-aymeric.mcrae@mail.mcgill.ca',
      license='MIT',
      install_requires=[
          'torch'
      ],
      packages=['cgoptimizer'],
      zip_safe=False)
