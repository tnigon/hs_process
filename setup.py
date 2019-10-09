# -*- coding: utf-8 -*-

import setuptools

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

#def history():
#    with open('HISTORY.md') as history_file:
#        return history_file.read()

requirements = [
#    'ast',
#    'itertools',
#    'json',
#    'matplotlib',
#    'math',
#    'numpy',
#    'osgeo',
#    'pandas',
#    'PIL',
#    're',
    'spectral',
]
test_requirements = [
    # TODO: put package test requirements here
]

setuptools.setup(name='hs_process',
                 version='0.0.1',
                 description=('Tools for processing, manipulating, and '
                              'analyzing aerial hyperspectral imagery'),
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
#                package_data={'hyperspectral': ['examples/*', 'examples/data/*']},
#                include_package_data=True,
                install_requires=requirements,
#                test_suite='tests',
#                tests_require=test_requirements,
                zip_safe=False)
