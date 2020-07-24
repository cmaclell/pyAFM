from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()

long_description = '...need to add description'

setup(
    name='pyafm',
    version='0.1.0',
    author='Christopher J MacLellan',
    author_email='maclellan.christopher@gmail.com',
    url='https://github.com/cmaclell/pyafm',
    description='A python implementation of the additive factors model that provides utilities for fitting and plotting learning curves.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=find_packages(),
    # scripts=['bin/altrain'],
    # entry_points ={

    # },
        entry_points={
            "console_scripts": [
                "afmfit = pyafm.process_datashop:main",
                "afmplot = pyafm.plot_datashop:main"
            ]
    },
    classifiers=(
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
    ),
    keywords='learning curve additive factors model',
    install_requires=requirements,
    zip_safe=False
)
