from setuptools import setup

setup(name='cgoptimizer',
      version='0.1',
      description='Critical Gradient Opimizaters',
      url='https://github.com/chandar-lab/CGOptimizer',
      author='Paul-Aymeric McRae',
      author_email='paul-aymeric.mcrae@mail.mcgill.ca',
      license='MIT',
	  install_requires=[
          'torch',
		  'torchvision'
      ],
      packages=['cgoptimizer'],
      zip_safe=False)