# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:24:54 2020

@author: nigo0024
"""

'''
Set up a test to run through all of the functions (and various options) and
provide a report on how many were returned with errors (coverage).

Then this file can be run anytime a change is made and changes are pushed to
see if anything was broken
'''

import os
import unittest
import spectral.io.spyfile as SpyFile

from hs_process import hsio


class Test_hsio_read_cube(unittest.TestCase):
    def setUp(self):
        '''
        This setUp function will be called for every single test that is run
        '''
        self.fname_hdr = fname_hdr
        self.io = hsio(fname_hdr)

    def tearDown(self):
        '''
        This tearDown function will be called after each test method is run
        '''
        self.fname_hdr = None
        self.io = None

    def test_names(self):
        self.assertEqual(self.io.name_long, '-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum',
                         'Incorrect long name determination')
        self.assertEqual(self.io.name_short, 'Wells_rep2_20180628_16h56m_pika_gige_7',
                         'Incorrect short name determination')
        self.assertEqual(self.io.name_plot, '7',
                         'Incorrect plot name determination')

    def test_readability(self):
        io1 = hsio()
        io1.read_cube(fname_hdr)
        self.assertIsInstance(io1.spyfile, SpyFile.SpyFile,
                              'Not a SpyFile object')

        io2 = hsio()
        io2.read_cube(os.path.splitext(fname_hdr)[0])
        self.assertIsInstance(io2.spyfile, SpyFile.SpyFile,
                              'Not a SpyFile object')

        io3 = hsio(fname_hdr)
        self.assertIsInstance(io3.spyfile, SpyFile.SpyFile,
                              'Not a SpyFile object')

        io4 = hsio(os.path.splitext(fname_hdr)[0])
        self.assertIsInstance(io4.spyfile, SpyFile.SpyFile,
                              'Not a SpyFile object')

    @unittest.skip("demonstrating skipping")
    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    @unittest.expectedFailure
    def test_failure(self):
        '''
        Mark the test as an expected failure. If the test fails it will be
        considered a success. If the test passes, it will be considered a
        failure.
        '''
        assert 1 == 2


class Test_hsio_read_spec(unittest.TestCase):
    def setUp(self):
        '''
        This setUp function will be called for every single test that is run
        '''
        self.io = hsio()
        self.io.read_spec(fname_hdr_spec)

    def tearDown(self):
        '''
        This tearDown function will be called after each test method is run
        '''
        self.io = None

    def test_names(self):
        self.assertEqual(self.io.name_long, '-cube-to-spec-mean',
                         'Incorrect long name determination')
        self.assertEqual(self.io.name_short, 'Wells_rep2_20180628_16h56m_pika_gige_7_plot_611',
                         'Incorrect short name determination')
        self.assertEqual(self.io.name_plot, '611',
                         'Incorrect plot name determination')

    def test_readability(self):
        io1 = hsio()
        io1.read_spec(fname_hdr_spec)
        self.assertIsInstance(io1.spyfile_spec, SpyFile.SpyFile,
                              'Not a SpyFile object')

        io2 = hsio()
        io2.read_spec(os.path.splitext(fname_hdr_spec)[0])
        self.assertIsInstance(io2.spyfile_spec, SpyFile.SpyFile,
                              'Not a SpyFile object')


class Test_hsio_set_io_defaults(unittest.TestCase):
    def setUp(self):
        '''
        This setUp function will be called for every single test that is run
        '''
        self.io = hsio()

    def tearDown(self):
        '''
        This tearDown function will be called after each test method is run
        '''
        self.io = None

    def test_dtype(self):
        self.io.set_io_defaults(dtype=int)
        self.assertEqual(self.io.defaults.envi_write.dtype, int,
                         'dtype not properly modified')
    def test_force(self):
        self.io.set_io_defaults(force=True)
        self.assertEqual(self.io.defaults.envi_write.force, True,
                         'force not properly modified')
    def test_ext(self):
        self.io.set_io_defaults(ext='spec')
        self.assertEqual(self.io.defaults.envi_write.ext, 'spec',
                         'ext not properly modified')
    def test_interleave(self):
        self.io.set_io_defaults(interleave='bsq')
        self.assertEqual(self.io.defaults.envi_write.interleave, 'bsq',
                         'interleave not properly modified')
    def test_byteorder(self):
        self.io.set_io_defaults(byteorder=1)
        self.assertEqual(self.io.defaults.envi_write.byteorder, 1,
                         'byteorder not properly modified')


def suite():
    suite = unittest.TestSuite()

    # read_cube
    suite.addTest(Test_hsio_read_cube('test_readability'))
    suite.addTest(Test_hsio_read_cube('test_names'))
    suite.addTest(Test_hsio_read_cube('test_split'))
    suite.addTest(Test_hsio_read_cube('test_failure'))

    # read_spec
    suite.addTest(Test_hsio_read_spec('test_readability'))
    suite.addTest(Test_hsio_read_spec('test_names'))

    # set_io_defaults
    suite.addTest(Test_hsio_set_io_defaults('test_dtype'))
    suite.addTest(Test_hsio_set_io_defaults('test_force'))
    suite.addTest(Test_hsio_set_io_defaults('test_ext'))
    suite.addTest(Test_hsio_set_io_defaults('test_interleave'))
    suite.addTest(Test_hsio_set_io_defaults('test_byteorder'))
    return suite

if __name__ == '__main__':
    fname_hdr = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
    fname_hdr_spec = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7_plot_611-cube-to-spec-mean.spec.hdr'
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

