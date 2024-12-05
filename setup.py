from setuptools import setup, find_packages

setup(
    name='AutoTandemML',
    version='0.1.0',
    author='Luka Grbcic', 
    author_email='lgrbcic@lbl.gov',
    description='A package for automated active learning ehnanced tandem neural networks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lgrbcic/AutoTandemML', 
    packages=find_packages(where='src'), 
    package_dir={'': 'src'},
    install_requires=[
        'Indago==0.5.2',
        'scikit-learn==1.2.2',
        'scikit-learn-extra==0.3.0',
        'numpy==1.24.3',
        'xgboost==2.0.3',
        'scipy==1.11.4',
        'torch==2.3.0',
        
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
    ],
    python_requires='>=3.6',
)

