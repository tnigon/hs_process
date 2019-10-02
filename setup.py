# -*- coding: utf-8 -*-
"""
© 2019 Regents of the University of Minnesota. All rights reserved.
"""
__copyright__ = 'Regents of the University of Minnesota. All rights reserved.'
__author__ = 'Tyler Nigon'
__license__ = (
        '"hyperspectral" is copyrighted by the Regents of the University of Minnesota.'
        'It can be freely used for educational and research purposes by '
        'non-profit institutions and US government agencies only. Other '
        'organizations are allowed to use "envi_crop" only for evaluation '
        'purposes, and any further uses will require prior approval. The '
        'software may not be sold or redistributed without prior approval. '
        'One may make copies of the software for their use provided that the '
        'copies are not sold or distributed, are used under the same terms '
        'and conditions.'
        'As unestablished research software, this code is provided on an '
        '"as is" basis without warranty of any kind, either expressed or '
        'implied. The downloading, or executing any part of this software '
        'constitutes an implicit agreement to these terms. These terms and '
        'conditions are subject to change at any time without prior notice.')
__email__ = 'nigo0024@umn.edu'

from setuptools import setup

setup(name='hyperspectral',
      version='0.0.1',
      description=('Tools for processing, manipulating, and analyzing aerial '
                   'hyperspectral imagery'),
      url='https://github.com/tnigon/hyperspectral',
      author='Tyler J. Nigon',
      author_email='nigo0024@umn.edu',
#      license='MIT',
      packages=['hyperspectral'],
      zip_safe=False)