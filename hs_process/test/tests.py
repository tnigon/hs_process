# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:07:59 2020

@author: nigo0024
"""

import unittest

import test_hsio

fname_hdr = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
fname_hdr_spec = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7_plot_611-cube-to-spec-mean.spec.hdr'
runner = unittest.TextTestRunner(verbosity=2)
runner.run(test_hsio.suite())
