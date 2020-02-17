# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:07:59 2020

@author: nigo0024

This test runner allows me to load in each of the test modules, then load all
the tests from each of those modules into a test suite before runnning them all
"""

import unittest

import test_hsio

# initialize
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(test_hsio))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

#fname_hdr = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
#fname_hdr_spec = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7_plot_611-cube-to-spec-mean.spec.hdr'
#runner = unittest.TextTestRunner(verbosity=2)
#runner.run(test_hsio.suite())
