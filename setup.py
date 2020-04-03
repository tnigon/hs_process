# -*- coding: utf-8 -*-

import setuptools

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

requirements = [
# 'gdal',
 'geopandas',  # must be installed with Anaconda because of shapely dependency
# 'numpy',
# 'pandas',
 'seaborn',
# 'matplotlib',
 'spectral',  # segment
 ]

test_requirements = [
    # TODO: put package test requirements here
]

setuptools.setup(name='hs_process',
                 version='0.0.4',
                 description=('An open-source Python package for geospatial '
                              'processing of aerial hyperspectral imagery'),
                 long_description=readme(),
                 long_description_content_type="text/markdown",
                 url='https://github.com/tnigon/hyperspectral',
                 author='Tyler J. Nigon',
                 author_email='nigo0024@umn.edu',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 classifiers=[
                         'Development Status :: 4 - Beta',
                         'Intended Audience :: Science/Research',
                         'Natural Language :: English',
                         'Operating System :: Microsoft :: Windows',
                         'Programming Language :: Python :: 3',
                         ],
                package_data={'hs_process': ['examples/*', 'tests/*', 'data/*']},
                include_package_data=True,
                install_requires=requirements,
#                test_suite='hs_process/tests',
                tests_require=test_requirements,
                zip_safe=False)
