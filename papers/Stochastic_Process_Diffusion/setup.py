from setuptools import setup, find_packages

install_requires = [
    'numpy>=1.21.5',
    'pytest>=6.2.4',
    'scipy>=1.7.1',
    'torch>=1.12.1',
    'pytorch-lightning==1.6.0',
    'torchdiffeq==0.2.3',
    'torchsde==0.2.5',
    'matplotlib>=3.4.3',
    'seaborn==0.11.1',
    'pytorchts==0.6.0',
    'gluonts==0.9.*',
    'wget==3.2',
]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='tsdiff',
      version='0.1.0',
      description='Time series diffusion',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='',
      author='Marin Bilos',
      author_email='marin.bilos@morganstaley.com', # also: marin.bilos@tum.de
      packages=find_packages(),
      install_requires=install_requires,
      python_requires='>=3.7',
      zip_safe=False,
)
