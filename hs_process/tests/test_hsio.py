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

The following test uses a very small "test" datacube that is only 3x3x240
(8,640 bytes)
'''
import numpy as np
import os
import shutil, tempfile
import spectral.io.spyfile as SpyFile
import unittest

from hs_process import hsio
from hs_process import spatial_mod

NAME_CUBE = 'Wells_rep2_20180628_16h56m_test_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
NAME_SPEC = 'Wells_rep2_20180628_16h56m_pika_gige_7_plot_611-cube-to-spec-mean.spec.hdr'
FILENAME_HDR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', NAME_CUBE)
FILENAME_HDR_SPEC = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', NAME_SPEC)
# print(FILENAME_HDR)
if not os.path.isfile(FILENAME_HDR):
    FILENAME_HDR = os.path.join(os.path.dirname(os.getcwd()), 'data', NAME_CUBE)
if not os.path.isfile(FILENAME_HDR_SPEC):
    FILENAME_HDR_SPEC = os.path.join(os.path.dirname(os.getcwd()), 'data', NAME_SPEC)

class Test_hsio_get_fname_hdr(unittest.TestCase):
    def setUp(self):
        '''
        This setUp function will be called for every single test that is run
        '''
        self.fname_hdr = FILENAME_HDR
        self.io = hsio(FILENAME_HDR)

    def tearDown(self):
        '''
        This tearDown function will be called after each test method is run
        '''
        self.fname_hdr = None
        self.io = None

    def test_bip_extensions(self):
        fname_bip_hdr = os.path.join(self.io.base_dir, 'image_name_out.bip.hdr')
        fname_hdr = os.path.join(self.io.base_dir, 'image_name_out.hdr')
        fname_bip = os.path.join(self.io.base_dir, 'image_name_out.bip')
        fname_bip_hdr_correct = os.path.join(self.io.base_dir, 'image_name_out.bip.hdr')

        self.assertEqual(fname_bip_hdr_correct, self.io._get_fname_hdr(fname_bip_hdr),
                         'hsio._get_fname_hdr() ".bip.hdr" is incorrect')
        self.assertEqual(fname_bip_hdr_correct, self.io._get_fname_hdr(fname_hdr),
                         'hsio._get_fname_hdr() ".hdr" is incorrect')
        self.assertEqual(fname_bip_hdr_correct, self.io._get_fname_hdr(fname_bip),
                         'hsio._get_fname_hdr() ".bip" is incorrect')

    def test_spec_extensions(self):
        fname_spec_hdr = os.path.join(self.io.base_dir, 'image_name_out.spec.hdr')
        fname_spec = os.path.join(self.io.base_dir, 'image_name_out.spec')
        fname_spec_hdr_correct = os.path.join(self.io.base_dir, 'image_name_out.spec.hdr')

        self.assertEqual(fname_spec_hdr_correct, self.io._get_fname_hdr(fname_spec_hdr),
                         'hsio._get_fname_hdr() ".spec.hdr" is incorrect')
        self.assertEqual(fname_spec_hdr_correct, self.io._get_fname_hdr(fname_spec),
                         'hsio._get_fname_hdr() ".spec" is incorrect')

class Test_hsio_read_cube(unittest.TestCase):
    def setUp(self):
        '''
        This setUp function will be called for every single test that is run
        '''
        self.fname_hdr = FILENAME_HDR
        self.io = hsio(FILENAME_HDR)

    def tearDown(self):
        '''
        This tearDown function will be called after each test method is run
        '''
        self.fname_hdr = None
        self.io = None

    def test_names(self):
        self.assertEqual(self.io.name_long, '-Convert Radiance Cube to Reflectance from Measured Reference Spectrum',
                         'Incorrect long name determination')
        self.assertEqual(self.io.name_short, 'Wells_rep2_20180628_16h56m_test_pika_gige_7',
                         'Incorrect short name determination')
        self.assertEqual(self.io.name_plot, '7',
                         'Incorrect plot name determination')

    def test_readability(self):
        io1 = hsio()
        io1.read_cube(self.fname_hdr)
        self.assertIsInstance(io1.spyfile, SpyFile.SpyFile,
                              'Not a SpyFile object')

        io2 = hsio()
        io2.read_cube(os.path.splitext(self.fname_hdr)[0])
        self.assertIsInstance(io2.spyfile, SpyFile.SpyFile,
                              'Not a SpyFile object')

        io3 = hsio(self.fname_hdr)
        self.assertIsInstance(io3.spyfile, SpyFile.SpyFile,
                              'Not a SpyFile object')

        io4 = hsio(os.path.splitext(self.fname_hdr)[0])
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
        self.fname_hdr_spec = FILENAME_HDR_SPEC
        self.io = hsio()
        self.io.read_spec(FILENAME_HDR_SPEC)

    def tearDown(self):
        '''
        This tearDown function will be called after each test method is run
        '''
        self.fname_hdr_spec = None
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
        io1.read_spec(self.fname_hdr_spec)
        self.assertIsInstance(io1.spyfile_spec, SpyFile.SpyFile,
                              'Not a SpyFile object')

        io2 = hsio()
        io2.read_spec(os.path.splitext(self.fname_hdr_spec)[0])
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

    def test_byteorder(self):
        self.io.set_io_defaults(byteorder=1)
        self.assertEqual(self.io.defaults.envi_write.byteorder, 1,
                         'byteorder not properly modified')
    def test_dtype(self):
        self.io.set_io_defaults(dtype=int)
        self.assertEqual(self.io.defaults.envi_write.dtype, int,
                         'dtype not properly modified')
    def test_ext(self):
        self.io.set_io_defaults(ext='spec')
        self.assertEqual(self.io.defaults.envi_write.ext, 'spec',
                         'ext not properly modified')
    def test_force(self):
        self.io.set_io_defaults(force=True)
        self.assertEqual(self.io.defaults.envi_write.force, True,
                         'force not properly modified')
    def test_interleave(self):
        self.io.set_io_defaults(interleave='bsq')
        self.assertEqual(self.io.defaults.envi_write.interleave, 'bsq',
                         'interleave not properly modified')
    def test_instance_independence(self):
        self.io.set_io_defaults(dtype=int, force=True, ext='spec',
                                interleave='bsq', byteorder=1)
        io2 = hsio()
        self.assertNotEqual(self.io.defaults.envi_write.ext,
                            io2.defaults.envi_write.ext,
                            ('New instance of hsio did not result in a new'
                             'instance of defaults'))
    def test_return_values(self):
        self.io.set_io_defaults(dtype=np.float32, force=False, ext='',
                                interleave='bip', byteorder=0)
        self.assertEqual(self.io.defaults.envi_write.dtype, np.float32,
                         'dtype not properly returned')
        self.assertEqual(self.io.defaults.envi_write.force, False,
                         'force not properly returned')
        self.assertEqual(self.io.defaults.envi_write.ext, '',
                         'ext not properly returned')
        self.assertEqual(self.io.defaults.envi_write.interleave, 'bip',
                         'interleave not properly returned')
        self.assertEqual(self.io.defaults.envi_write.byteorder, 0,
                         'byteorder not properly returned')


class Test_hsio_write_cube(unittest.TestCase):
    def setUp(self):
        '''
        This setUp function will be called for every single test that is run
        '''
        self.fname_hdr = FILENAME_HDR
        self.test_dir = tempfile.mkdtemp()
        self.io = hsio(FILENAME_HDR)
        self.my_spatial_mod = spatial_mod(self.io.spyfile)

    def tearDown(self):
        '''
        This tearDown function will be called after each test method is run
        '''
        self.fname_hdr = None
        self.io = None
        self.my_spatial_mod = None
        self.metadata = None
        shutil.rmtree(self.test_dir)
        self.test_dir = None

    def test_write_cube(self):
        fname_hdr_out = os.path.join(self.test_dir, self.io.name_short +
                                     '.bip.hdr')
        self.io.write_cube(fname_hdr_out, self.io.spyfile.open_memmap(),
                           metadata=self.io.spyfile.metadata)
        io2 = hsio(fname_hdr_out)
        self.assertIsInstance(io2.spyfile, SpyFile.SpyFile,
                              'Not a SpyFile object')


class Test_hsio_write_spec(unittest.TestCase):
    def setUp(self):
        '''
        This setUp function will be called for every single test that is run
        '''
        self.fname_hdr = FILENAME_HDR
        self.test_dir = tempfile.mkdtemp()
        self.io = hsio(FILENAME_HDR)
        self.spec_mean, self.spec_std, _ = self.io.tools.mean_datacube(self.io.spyfile.open_memmap())

    def tearDown(self):
        '''
        This tearDown function will be called after each test method is run
        '''
        self.fname_hdr = None
        self.io = None
        self.spec_mean = None
        self.spec_std = None
        shutil.rmtree(self.test_dir)

    def test_write_spec(self):
        fname_hdr_spec = os.path.join(self.test_dir, self.io.name_short +
                                      '-mean.spec.hdr')
        self.io.write_spec(fname_hdr_spec, self.spec_mean, self.spec_std)
        io2 = hsio()
        io2.read_spec(fname_hdr_spec)
        self.assertIsInstance(io2.spyfile_spec, SpyFile.SpyFile,
                              'Not a SpyFile object')

class Test_hsio_write_tif(unittest.TestCase):
    def setUp(self):
        '''
        This setUp function will be called for every single test that is run
        '''
        self.test_dir = tempfile.mkdtemp()
        self.fname_hdr = FILENAME_HDR
        self.io = hsio(FILENAME_HDR)
        self.my_spatial_mod = spatial_mod(self.io.spyfile)

    def tearDown(self):
        '''
        This tearDown function will be called after each test method is run
        '''
        self.fname_hdr = None
        self.io = None
        self.my_spatial_mod = None
        self.spec_std = None
        shutil.rmtree(self.test_dir)

    def test_write_tif_multi(self):
        '''
        `show_img` is only expected to work in an IPython console
        '''
        fname_tif = os.path.join(self.test_dir, self.io.name_short +
                                 '.tif')
        self.io.write_tif(fname_tif, self.io.spyfile.open_memmap(),
                          fname_in=self.fname_hdr, show_img=False)
        # print(os.path.getsize(fname_tif))
        self.assertGreater(os.path.getsize(fname_tif), 470,  # size should be 478
                         'Geotiff is not the correct size.')

    def test_write_tif_single(self):
        '''
        `show_img` is only expected to work in an IPython console
        '''
        fname_tif = os.path.join(self.test_dir, self.io.name_short +
                                 '.tif')
        self.io.write_tif(fname_tif, self.io.spyfile.open_memmap()[:,:,0],
                          fname_in=self.fname_hdr, show_img=False)
        # print(os.path.getsize(fname_tif))
        self.assertGreater(os.path.getsize(fname_tif), 380,  # size should be 382
                          'Geotiff is not the correct size.')

def suite():
    suite = unittest.TestSuite()

    # _get_fname_hdr
    suite.addTest(Test_hsio_get_fname_hdr('test_bip_extensions'))
    suite.addTest(Test_hsio_get_fname_hdr('test_spec_extensions'))


    # read_cube
    suite.addTest(Test_hsio_read_cube('test_readability'))
    suite.addTest(Test_hsio_read_cube('test_names'))
    suite.addTest(Test_hsio_read_cube('test_split'))
    suite.addTest(Test_hsio_read_cube('test_failure'))

    # read_spec
    suite.addTest(Test_hsio_read_spec('test_readability'))
    suite.addTest(Test_hsio_read_spec('test_names'))

    # set_io_defaults
    suite.addTest(Test_hsio_set_io_defaults('test_byteorder'))
    suite.addTest(Test_hsio_set_io_defaults('test_dtype'))
    suite.addTest(Test_hsio_set_io_defaults('test_ext'))
    suite.addTest(Test_hsio_set_io_defaults('test_force'))
    suite.addTest(Test_hsio_set_io_defaults('test_interleave'))
    suite.addTest(Test_hsio_set_io_defaults('test_instance_independence'))
    suite.addTest(Test_hsio_set_io_defaults('test_return_values'))

    # write_cube
    suite.addTest(Test_hsio_write_cube('test_write_cube'))

    # write_spec
    suite.addTest(Test_hsio_write_spec('test_write_spec'))

    # write_tif
    suite.addTest(Test_hsio_write_tif('test_write_tif_multi'))
    suite.addTest(Test_hsio_write_tif('test_write_tif_single'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

