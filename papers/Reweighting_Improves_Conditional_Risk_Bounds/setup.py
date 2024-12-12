from setuptools import find_packages, setup

install_requires = [
    'click==8.1.7',
    'matplotlib==3.9.2',
    'numpy==1.26.4',
    'pytorch_lightning==2.4.0',
    'PyYAML==6.0.2',
    'scikit_learn==1.5.2',
    'scipy==1.14.1',
    'tensorboard==2.18.0',
    'torch==2.2.2+cu121',
    'torchmetrics==1.4.1',
]

with open('README.md', 'r') as f:
    long_description = f.read()
    
setup(
    name='weighted_erm',
    version='0.1.0',
    description='',
    long_description=long_description,
    context_type='text/markdown',
    url='',
    author='Yikai Zhang, Jiahe Lin, et al.',
    author_email='jiahe.lin@morganstanley.com, yikai.zhang@morganstanley.com',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.10.1',
    zip_safe=False
)
