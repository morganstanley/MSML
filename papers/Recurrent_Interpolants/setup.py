from setuptools import find_packages, setup

install_requires = [
    'matplotlib==3.8.2',
    'numpy==1.26.2',
    'pandas==1.5.3',
    'scipy==1.10.1',
    'scikit-learn==1.2.2',
    'statsmodels==0.14.4'
    'POT==0.9.3',
    'torch~=2.2.1',
    'torchsde==0.2.5',
    'gluonts==0.14.3',
    'pytorchts @ git+https://github.com/zalandoresearch/pytorch-ts.git@e359530a11c13e34fb57f6ceaff6542e57205e50',
    'lightning @ git+https://github.com/Lightning-AI/lightning@6cbe9ceb560d798892bdae9186291acf9bf5d2e3',
]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='time_match',
    version='0.1.0',
    description='Recurrent Interpolants for Probabilistic Time Series Prediction',
    long_description=long_description,
    content_type='text/markdown',
    url='',
    author='Morgan Stanley Machine Learning',
    author_email='msml-qa@morganstanley.com',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.11.1',
    zip_safe=False,
)
