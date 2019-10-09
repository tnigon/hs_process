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
        'organizations are allowed to use "hyperspectral" only for evaluation '
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

import ast
import itertools
import json
from matplotlib import pyplot as plt
from math import factorial
import numpy as np
from osgeo import gdal
from osgeo import gdalconst
from osgeo import ogr
import pandas as pd
from PIL import Image
import os
import re
import spectral.io.envi as envi
import sys


class Hyperspectral(object):
    '''
    Performs band math on and ENVI image file
    # 0: read image
    # 1: add "byte order = 0" and "header offset = 0" to each .hdr file
    # 2: given ul pixel and number of pixels, crop image as array
    # 3: save cropped image to newfile as .bsq --> only option [I think] if
        GDAL requires to save array a single band at a time..
    # 4: modify .hdr with band information
    '''

    def __init__(self):
        '''
        '''
        self.fname_in = None
        self.base_dir = None
        self.fname_shp = None
        self.long_name = None
        self.base_dir_out = None

        self.img_ds = None
        self.img_sp = None
        self.array_smooth = None
        self.mask_array_3d = None
        self.meta_bands = None
        self.metadata = None
        self.plot = None
        self.geotransform = None
        self.projection = None
        self.ul_x_m = None
        self.ul_y_m = None
        self.size_x_m = None  # get from .hdr
        self.size_y_m = None
        self.buf_x_pix = None
        self.buf_y_pix = None
        self.spec_clip = None

        self.name_short = None
        self.name_long = None
        self.name_plot = None

        self.pix_skip = int(6.132 / -0.04)  # alley - skip 6.132 m
        self.cols_plots = None
        self.rows_plots = None
        self.row_plots_top = 0
        self.row_plots_bot = 0
        self.df_plots = pd.DataFrame(columns=['plot_id', 'col_plot',
                                              'row_plot', 'col_pix',
                                              'row_pix'])
        self.df_plots_single = pd.DataFrame(columns=['directory', 'name_short',
                                                     'name_long', 'easting_pix',
                                                     'northing_pix', 'buffer-x',
                                                     'buffer-y'])
        self.df_shp = pd.DataFrame(columns=['plot_id', 'ul_x_utm', 'ul_y_utm'])

#    def _read_envi_gdal(self):
#        '''
#        Converts a binary file of ENVI type to a numpy array.
#        Lack of an ENVI .hdr file will cause this to crash.
#        '''
#        drv = gdal.GetDriverByName('ENVI')
#        drv.Register()
#        img_ds = gdal.Open(self.fname_in, gdalconst.GA_ReadOnly)
#        if img_ds is None:
#            sys.exit("Image not loaded. Check file path and try again.")
#
#        self.geotransform = img_ds.GetGeoTransform()
#        self.projection = img_ds.GetProjection()
#        self.ul_x_m = self.geotransform[0]
#        self.ul_y_m = self.geotransform[3]
#        self.size_x_m = self.geotransform[1]
#        self.size_y_m = self.geotransform[5]
#
#        md_str = img_ds.GetMetadata_Dict()
#        meta_bands = {}
#        for i in range(len(md_str)):
#            try:
#                meta_bands[i+1] = float(md_str['Band_' + str(i+1)])
#            except ValueError as e:
#                temp = md_str['Band_' + str(i+1)]
#                meta_bands[i+1] = float(temp[temp.find("(")+1:temp.find(")")])
#        self.meta_bands = meta_bands
#        self.img_ds = img_ds

    def _read_envi_hdr(self, fname_in=None):
        '''
        Reads ENVI .hdr file and
        '''
        if fname_in is None:
            fname_in = self.fname_in
        if os.path.isfile(fname_in + '.hdr'):
            fname_hdr = fname_in + '.hdr'
        else:
            fname_hdr = fname_in[:-4] + '.hdr'
        with open(fname_hdr, 'r') as f:
            data = f.readlines()
        matches = []
        regex1 = re.compile(r'^(.+?)\s*=\s*({\s*.*?\n*.*?})$', re.M | re.I)
        regex2 = re.compile(r'^(.+?)\s*=\s*(.*?)$', re.M | re.I)
        for line in data:
            matches.extend(regex1.findall(line))
            subhdr = regex1.sub('', line)  # remove from line
            matches.extend(regex2.findall(subhdr))
        self.metadata = dict(matches)

        meta_bands = {}
        if 'band names' not in self.metadata.keys():
            for key, val in enumerate(sorted(ast.literal_eval(self.metadata['wavelength']))):
                meta_bands[key+1] = val
        else:
            try:
                band_names = list(sorted(ast.literal_eval(self.metadata['band names'])))
                wl_names = list(sorted(ast.literal_eval(self.metadata['wavelength'])))
            except ValueError as e:
                band_names = list(sorted(ast.literal_eval(str(self.metadata['band names']))))
                wl_names = list(sorted(ast.literal_eval(str(self.metadata['wavelength']))))
            for idx in range(len(band_names)):
                meta_bands[band_names[idx]] = wl_names[idx]
        self.meta_bands = meta_bands

    def _read_envi_spy(self):
        '''
        Reads ENVI file using Spectral Python; more streamlined features
        for hyperspectral manipulation, classification, and data display
        '''
        meta = self.metadata
        if 'byte order' not in meta.keys():
            meta['byte order'] = 0
            self._append_hdr('byte order', 0)
        # Note: img_sp.asarray() is always in .bsq order (x, y, z)
        self.img_sp = envi.open(self.fname_in + '.hdr')

        try:
            self.ul_y_m = float(self.img_sp.metadata['map info'][3])
            self.ul_x_m = float(self.img_sp.metadata['map info'][4])
            self.size_x_m = float(self.img_sp.metadata['map info'][5])
            self.size_y_m = float(self.img_sp.metadata['map info'][6])
        except KeyError as e:
            self.ul_y_m = None
            self.ul_x_m = None
            self.size_x_m = None
            self.size_y_m = None

    def _savitzky_golay(self, y, window_size=5, order=2, deriv=0, rate=1):
        '''
        Smooth (and optionally differentiate) data with a Savitzky-Golay
        filter. The Savitzky-Golay filter removes high frequency noise from
        data. It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.

        Parameters:
        y (`numpy.array`; shape (N,)): the values of the time history of the
            signal.
        window_size (`int`): the length of the window; must be an odd integer
            number (default: 5).
        order (`int`): the order of the polynomial used in the filtering; must
            be less than `window_size` - 1 (default: 2).
        deriv (`int`): the order of the derivative to compute (default: 0,
              means only smoothing).

        Returns:
        ys (`ndarray`; shape (N)): the smoothed signal (or it's n-th
           derivative).

        Notes:
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.

        Examples:
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        '''
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError as msg:
            raise ValueError('Window_size/order have to be of type int')
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError('Window_size must be a positive odd number')
        if window_size < order + 2:
            raise TypeError('Window_size is too small for the polynomials '
                            'order')
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with values taken from the signal
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')

    def _smooth_image(self, array=None, window_size=19, order=2):
        '''
        Applies the Savitzky Golay smoothing algorithm to the spectral
        domain of each image pixel
        '''
        if array is None:
            array = self.img_sp.asarray()
        array_2d = array.reshape(array.shape[0]*array.shape[1], array.shape[2])
        array_2d_temp = array_2d.copy()
        for idx, row in enumerate(array_2d):
            array_2d_temp[idx, :] = self._savitzky_golay(
                    row, window_size=window_size, order=order)
#            sns.lineplot(list(hs.meta_bands.keys()), array_2d[1000])
#            sns.lineplot(list(hs.meta_bands.keys()), array_2d_temp[1000])
        return array_2d_temp.reshape((array.shape))

    def _read_plot_shp(self):
        '''
        Reads shapefile of plot bounds and record upper left (northwest)
        corner of each plot
        '''
        df_shp = self.df_shp.copy()
        drv = ogr.GetDriverByName('ESRI Shapefile')
        ds_shp = drv.Open(self.fname_shp, 0)
        if ds_shp is None:
            print('Could not open {0}'.format(self.fname_shp))
        layer = ds_shp.GetLayer()

        for feat in layer:
            geom = feat.GetGeometryRef()
            bounds = geom.GetBoundary()
            bounds_dict = json.loads(bounds.ExportToJson())
            bounds_coords = bounds_dict['coordinates']
            plot_id = feat.GetField('plot')
            x, y = zip(*bounds_coords)
            ul_x_utm = min(x)
            ul_y_utm = max(y)
            df_temp = pd.DataFrame(data=[[plot_id, ul_x_utm, ul_y_utm]],
                                   columns=df_shp.columns)
            df_shp = df_shp.append(df_temp, ignore_index=True)
            self.df_shp = df_shp

    def _check_alley(self):
        '''
        Calculates whether there is an alleyway in the image (based on plot
        configuration), then adjusts rows_plots so it is correct after
        considering the alley
        '''

        plot_id_tens = abs(self.plot_id_ul) % 100
        self.row_plots_top = plot_id_tens % 9
        if self.row_plots_top == 0:
            self.row_plots_top = self.rows_plots  # remainder is 0, not 9..

        if self.row_plots_top < self.rows_plots:
            # get pix left over
            pix_remain = (self.img_sp.nrows - abs(self.ul_y_pix) -
                          (self.row_plots_top * abs(self.plot_y_pix)))
        else:
            return

        if pix_remain >= abs(self.pix_skip + self.plot_y_pix):
            # have room for more plots (must still remove 2 rows of plots)
            # calculate rows remain after skip
            self.row_plots_bot = int(abs((pix_remain + self.pix_skip) /
                                     self.plot_y_pix))
            self.rows_plots = self.row_plots_top + self.row_plots_bot
        elif pix_remain >= abs(self.plot_y_pix) * 2:
            # remove 2 rows of plots
            self.rows_plots -= 2
        elif pix_remain >= abs(self.plot_y_pix):
            # remove 1 row of plots
            self.rows_plots -= 1
        else:
            # works out perfect.. don't have to change anything
            pass

    def _get_envi_gdal(self, fname_in=None):
        '''
        GDAL dataset isn't saved to self, so this function gets and returns the
        GDAL object if necessary.
        '''
        if fname_in is None:
            fname_in = self.fname_in
        drv = gdal.GetDriverByName('ENVI')
        drv.Register()
        img_ds = gdal.Open(fname_in, gdalconst.GA_ReadOnly)
        if img_ds is None:
            sys.exit("Image not loaded. Check file path and try again.")
        return img_ds

    def _get_UTM(self, ulx, uly, utm_x, utm_y, size_x=0.04, size_y=-0.04):
        '''
        Calculates the new UTM coordinate of cropped plot
        '''
        utm_x_new = utm_x + (ulx * size_x)
        utm_y_new = utm_y - (uly * size_y)
        return utm_x_new, utm_y_new

    def _write_tif(self, array_img_crop, fname_out_tif, projection_out,
                   geotransform_out):
        '''
        Writes RGB geotif to file
        '''
        drv = gdal.GetDriverByName('GTiff')
        drv.Register()
        ysize, xsize, bands = array_img_crop.shape
        tif_out = drv.Create(fname_out_tif, xsize, ysize, 3, gdal.GDT_Float32)
        tif_out.SetProjection(projection_out)
        tif_out.SetGeoTransform(geotransform_out)

        band_b = self._get_band(460)[0]
        band_g = self._get_band(550)[0]
        band_r = self._get_band(640)[0]
        band_list = [band_b, band_g, band_r]

        array_img = None
        for idx, band in enumerate(band_list):
            # for whatever reason, GDAL needs N/S pixels (rows) to be flipped
            array_band = np.flip(array_img_crop[:, :, band-1], axis=0)
            band_out = tif_out.GetRasterBand(idx + 1)
            band_out.WriteArray(array_band)
            if array_img is None:
                array_img = array_band
            else:
                array_img = np.dstack((array_img, array_band))  # stacks bands
            band_out = None
        tif_out.FlushCache()
        drv = None
        tif_out = None
        self.show_img(array_img_crop, band_r=band_r, band_g=band_g, band_b=band_b)

    def _get_band(self, target):
        '''
        Returns band number of closest target wavelength
        band = self._get_band(703) returns 151 (i.e., band 151)

        Parameters:
            target (`int` or `float`): the target wavelength to retrive band
                number for (required).
        '''
        val_target = min(list(self.meta_bands.values()),
                         key=lambda x: abs(x-target))
        key_band = list(self.meta_bands.keys()
                        )[sorted(list(self.meta_bands.values())).index(val_target)]
        key_wavelength = sorted(list(self.meta_bands.values()))[key_band-1]
        return key_band, key_wavelength

    def _get_band_range(self, range_wl, index=True):
        '''
        Gets all band indexes with the given minimum and maximum wavelengths

        Parameters:
            range_wl (list): the minimum and maximum wavelength to consider;
                values should be `int` or `float`.
            index (bool): Indicates whether to return the band number (min=1)
                or to return index number (min=0) (default: True)
        '''
        band_min, wl_min = self._get_band(range_wl[0])
        band_max, wl_max = self._get_band(range_wl[1])
        if wl_min < range_wl[0]:
            band_min += 1
        if wl_max > range_wl[1]:
            band_max -= 1
        if index is True:
            band_min = self._get_band_index(band_min)
            band_max = self._get_band_index(band_max)
        band_list = [x for x in range(band_min, band_max+1)]
        return band_list

    def _write_envi(self, array, fname_out, geotransform_out, name=None,
                    interleave='bip', rewrite_hdr=True):
        '''
        Writes datacube to ENVI file

        Parameters:
            array (numpy array): input image cube; must be in band sequential
                (x, y, z) format to properly save base on interleave indicated
            fname_out (`str`):
            geotransform_out (`GDAL geotransform`):
            name (`str`):
            interleave (`str`):
            rewrite_hdr (`bool`): indicates if header file should be replaced
                by self.metadata items (default=True)
        '''
        try:
            ysize, xsize, bands = array.shape
        except ValueError as e:
            ysize, xsize = array.shape
            bands = 1

        base_name, ext = os.path.splitext(fname_out)
        if ext != '.' + interleave:
            fname_out = base_name + '.' + interleave

        if interleave.lower == 'bip':
            interleave_str = 'INTERLEAVE=BIP'
        elif interleave.lower == 'bil':
            interleave_str = 'INTERLEAVE=BIL'
        elif interleave.lower == 'bsq':
            interleave_str = 'INTERLEAVE=BSQ'

        drv = gdal.GetDriverByName('ENVI')
        drv.Register()
        ds_out = drv.Create(fname_out, xsize, ysize, bands, gdal.GDT_Float32,
                            ['SUFFIX=ADD', interleave_str])
        ds_out.SetGeoTransform(geotransform_out)
        ds_out.SetProjection(self.projection)

        for band in range(bands):
            if bands == 1:
                array_band = array[:, :]
            else:
                array_band = array[:, :, band]
            band_out = ds_out.GetRasterBand(band + 1)
            if name is None:
                band_out.SetDescription(str(band + 1))
            else:
                band_out.SetDescription(name)
            band_out.WriteArray(array_band)
            band_out = None
        ds_out.FlushCache()
        drv = None
        ds_out = None
        if rewrite_hdr is True:
            self._rewrite_hdr(fname_out)

    def _rewrite_hdr(self, fname_out_envi):
        '''
        Replaces .hdr file with self.metadata items
        '''
        _, ext = os.path.splitext(fname_out_envi)
        if ext != '.hdr':
            fname_out_envi += '.hdr'

        with open(fname_out_envi, 'w') as f:
            f.write('ENVI\n')
            for key, val in sorted(self.metadata.items()):
                f.write('{0} = {1}\n'.format(key, val))

    def _spec_hdr(self, fname_out):
        '''
        Appends "wavelength" info to .hdr file
        '''
#        basename, ext = os.path.splitext(fname_out)
#        fname_new = basename + ext + '.hdr'
#        os.rename(basename + '.hdr', fname_new)
#        fname_hdr = fname_new
        fname_hdr = fname_out + '.hdr'
        band_names = []
        wavelength = []
        for key, val in self.meta_bands.items():
            band_names.append(key)
            wavelength.append(val)
        band_names.sort()
        band_names_str = ', '.join(str(b) for b in band_names)
        band_names_str = '{' + band_names_str + '}'
        wavelength.sort()
        wavelength_str = ', '.join(str(wl) for wl in wavelength)
        wavelength_str = '{' + wavelength_str + '}'
        with open(fname_hdr, 'a') as f:
            f.write('{0} = {1}\n'.format('band names', band_names_str))
            f.write('{0} = {1}\n'.format('wavelength', wavelength_str))

    def _write_spec(self, array_index, fname_out, name=None):
        '''
        Writes datacube to ENVI file
        '''
        drv = gdal.GetDriverByName('ENVI')
        drv.Register()
        # Creates a new raster data source
        try:
            bands, ysize, xsize = array_index.shape
        except ValueError as e:
            ysize, xsize = array_index.shape
            bands = 1
        ds_out = drv.Create(fname_out, xsize, ysize, bands, gdal.GDT_Float32,
                            ['SUFFIX=ADD', 'INTERLEAVE=BIP'])
        for band in range(bands):
            if bands == 1:
                array_band = array_index[:, :]
            else:
                array_band = array_index[band, :, :]
            band_out = ds_out.GetRasterBand(band + 1)
            if name is None:
                band_out.SetDescription(str(band + 1))
            else:
                band_out.SetDescription(name)
            band_out.WriteArray(array_band)
            band_out = None
        ds_out.FlushCache()
        drv = None
        ds_out = None

    def _del_meta_item(self, my_dict, key):
        '''
        Deletes metadata item and returns new dictionary
        '''
        try:
            del my_dict[key]
        except KeyError:
            pass
        return my_dict

    def _write_envi_spy(self, fname_out, df_mean, df_std, interleave='bip',
                        dtype=np.float32, byteorder=0, force=True):
        '''
        Writes spectra to ENVI file
        '''
        if os.path.splitext(fname_out)[1] != '.hdr':
            fname_out = fname_out + '.hdr'

        metadata = self.metadata
        metadata = self._del_meta_item(metadata, 'map info')
        metadata = self._del_meta_item(metadata, 'history')

        metadata = self._del_meta_item(metadata, 'original cube file')
        metadata = self._del_meta_item(metadata, 'pointlist')
        metadata = self._del_meta_item(metadata, 'boundary')
        metadata = self._del_meta_item(metadata, 'label')

        band_names = ', '.join(str(e) for e in list(self.meta_bands.keys()))
        metadata['band names'] = '{' + band_names + '}'

        std = df_std.to_dict()
        stdev = ', '.join(str(e) for e in list(std.values()))
        metadata['stdev'] = '{' + stdev + '}'

        array_mean = df_mean.to_numpy()
        array = array_mean.reshape(1,1,len(df_mean))
        envi.save_image(fname_out, array, interleave=interleave, dtype=dtype,
                        byteorder=byteorder, metadata=metadata, force=force,
                        ext=None)

    def _write_spec_spy(self, fname_out, df_mean, df_std, interleave='bip',
                        dtype=np.float32, byteorder=0, force=True):
        '''
        Writes spectra to ENVI file
        '''
        if os.path.splitext(fname_out)[1] != '.hdr':
            fname_out = fname_out + '.hdr'

        metadata = self.metadata
        metadata = self._del_meta_item(metadata, 'map info')
        metadata = self._del_meta_item(metadata, 'history')

        metadata = self._del_meta_item(metadata, 'original cube file')
        metadata = self._del_meta_item(metadata, 'pointlist')
        metadata = self._del_meta_item(metadata, 'boundary')
        metadata = self._del_meta_item(metadata, 'label')

        band_names = ', '.join(str(e) for e in list(self.meta_bands.keys()))
        metadata['band names'] = '{' + band_names + '}'

        std = df_std.to_dict()
        stdev = ', '.join(str(e) for e in list(std.values()))
        metadata['stdev'] = '{' + stdev + '}'

        array_mean = df_mean.to_numpy()
        array = array_mean.reshape(1,1,len(df_mean))
        envi.save_image(fname_out, array, interleave=interleave, dtype=dtype,
                        byteorder=byteorder, metadata=metadata, force=force,
                        ext=None)

#imgs = Band_math(fname_in)

# create the output image
#driver = imgs.img_ds.GetDriver()
#print driver
#outDs = driver.Create(r"C:\Users\nigo0024\Downloads\reclass_40.tif", 100, 100, 1, gdal.GDT_Int32)
#if outDs is None:
#    print('Could not create reclass_40.tif')
#    sys.exit(1)
    def _get_band_info_consolidate(self, b1, b2):
        '''
        Gets band number and wavelength information
        '''
        if isinstance(b1, list):
            band1 = []
            wl1 = []
            for band in b1:
                band_i, wl_i = self._get_band(band)
                band1.append(band_i)
                wl1.append(wl_i)
            wl1 = np.mean(wl1)
        else:
            band1, wl1 = self._get_band(b1)
        if isinstance(b2, list):
            band2 = []
            wl2 = []
            for band in b2:
                band_i, wl_i = self._get_band(band)
                band2.append(band_i)
                wl2.append(wl_i)
            wl2 = np.mean(wl2)
        else:
            band2, wl2 = self._get_band(b2)
        return band1, band2, wl1, wl2

    def _get_band_index(self, band_num):
        '''
        Subtracts 1 from each number in band list and returns list of band
        indexes
        '''
        if isinstance(band_num, list):
            band_num = np.array(band_num)
            band_idx = list(band_num - 1)
        else:
            band_idx = band_num - 1
        return band_idx

    def _get_band_num(self, band_idx):
        '''
        Adds 1 to each number in band list and returns list of band
        numbers
        '''
        if isinstance(band_idx, list):
            band_idx = np.array(band_idx)
            band_num = list(band_idx + 1)
        else:
            band_num = band_idx + 1
        return band_num

    def _get_band_mean(self, img_array, band_num):
        '''
        Gets the mean value from a list of bands

        Parameters:
            img_array (numpy array): image to evaluate
            band_num (int, list): band number(s) to determine mean value for
        '''
        band_idx = self._get_band_index(band_num)
        if isinstance(band_idx, list):
            array_band = np.mean(img_array[:,:,band_idx], axis=2)
        else:
            array_band = img_array[:,:,band_idx]
        return array_band

    def _mask_array(self, array, thresh=0.55, side='lower'):
        '''
        Creates a masked numpy array based on a threshold value
        '''
        if side is 'lower':
            mask_array = np.ma.array(array, mask = array <= 0.55)
        elif side is 'upper':
            mask_array = np.ma.array(array, mask = array > 0.55)
        unmasked_pct = 100 * (mask_array.count()/
                              (array.shape[0]*array.shape[1]))
        print('Proportion unmasked pixels: {0:.2f}%'.format(unmasked_pct))
        return mask_array

    def _recurs_dir(self, base_dir, search_exp='.csv', level=None):
        '''
        Searches all folders and subfolders recursively within <base_dir>
        for filetypes of <search_exp>.
        Returns sorted <outFiles>, a list of full path strings of each result.

        Inputs:     Directory (base_dir) that includes files
                    (may be in subdirectories)
            base_dir: directory path that should include files to be returned
            search_exp: file format to search for in all directories
                    and subdirectories
                    Note: may include files with <search_exp> in name that have
                    different extensions.
            level: how many levels to search; if None, searches all levels
        Outputs:    out_files include the full pathname\filename\ext of all
                    files that have <search_exp> in their name.
        ===================================================================
        '''
        if level is None:
            level = 1
        else:
            level -= 1
        d_str = os.listdir(base_dir)
        out_files = []
        for item in d_str:
            full_path = os.path.join(base_dir, item)
            if not os.path.isdir(full_path) and item.endswith(search_exp):
                out_files.append(full_path)
            elif os.path.isdir(full_path) and level >= 0:
                new_dir = full_path  # If dir, then search in that
                out_files_temp = self._recurs_dir(new_dir, search_exp)
                if out_files_temp:  # if list is not empty
                    out_files.extend(out_files_temp)  # add items
        return sorted(out_files)

    def _append_hdr(self, key, value):
        '''
        Appends key and value to ENVI .hdr
        '''
        text_append = '{0} = {1}\n'.format(key, value)
        if os.path.isfile(self.fname_in + '.hdr'):
            fname_hdr = self.fname_in + '.hdr'
        else:
            fname_hdr = self.fname_in[:-4] + '.hdr'
        with open(fname_hdr, 'a') as f:
            f.write(text_append)

    def _xstr(self, s):
        if s is None:
            return ''
        return str('-' + s)

    def _save_file_setup(self, base_dir_out=None, folder_name='band_math'):
        '''
        Basic setup items when saving manipulated image files to disk

        Parameters:
            base_dir_out (`str`): The base
            folder_name (`str`):
        '''
        if base_dir_out is None:
            base_dir_out = os.path.join(self.base_dir, folder_name)
        if not os.path.isdir(base_dir_out):
            os.mkdir(base_dir_out)
        self.base_dir_out = base_dir_out

        if self.name_plot is not None:
            name_print = self.name_plot
        else:
            name_print = self.name_short
        return base_dir_out, name_print

    def _get_meta_set(self, meta_set, idx):
        '''
        Reads a value from meta_set based on the index

        Parameters:
            meta_set (`str`): the string representation of the metadata set
            idx (`int`): index to be read
        '''
        meta_set_list = meta_set[1:-1].split(",")
        meta_set_str = []
        for item in meta_set_list:
            if str(item)[::-1].find('.') == -1:
                try:
                    meta_set_str.append(int(item))
                except ValueError as e:
                    if item[0] is ' ':
                        meta_set_str.append(item[1:])
                    else:
                        meta_set_str.append(item)
            else:
                try:
                    meta_set_str.append(float(item))
                except ValueError as e:
                    if item[0] is ' ':
                        meta_set_str.append(item[1:])
                    else:
                        meta_set_str.append(item)
        return meta_set_str[idx]

    def _modify_meta_set(self, meta_set, idx, value):
        '''
        Modifies meta_set by converting string to list, then adjucting the
        value of an item by its index

        Parameters:
            meta_set (`str`): the string representation of the metadata set
            idx (`int`): index to be modified
            value (`float`, `int`, or `str`): value to replace at idx
        '''
        meta_set_list = meta_set[1:-1].split(",")
        meta_set_str = []
        for item in meta_set_list:
            if str(item)[::-1].find('.') == -1:
                try:
                    meta_set_str.append(int(item))
                except ValueError as e:
                    if item[0] is ' ':
                        meta_set_str.append(item[1:])
                    else:
                        meta_set_str.append(item)
            else:
                try:
                    meta_set_str.append(float(item))
                except ValueError as e:
                    if item[0] is ' ':
                        meta_set_str.append(item[1:])
                    else:
                        meta_set_str.append(item)
#            try:
#                meta_set_str.append(str(item))
#            except ValueError as e:
#                if item[0] is ' ':
#                    meta_set_str.append(item[1:])
#                else:
#                    meta_set_str.append(item)
        meta_set_str[idx] = str(value)
        set_str = '{'
        for i, item in enumerate(meta_set_str):
            set_str += str(item)
            if i+1 == len(meta_set_str):
                set_str += '}'
            else:
                set_str += ', '
        return set_str

    def read_cube(self, fname_in, fname_shp=None,
                  name_long=None, plot_name=False):
        '''
        Reads in a hyperspectral datacube

        fname_in (str): filename of datacube to be read
        fname_shp (str): filename of shapefile
        spectra_smooth (bool): If true, spectra for each pixel will be smoothed
        name_long (str): Spectronon processing appends processing names to
            the filenames; this indicates those processing names that are
            repetitive and can be deleted from the filename following
            processing.
        plot_name (bool): Indicates whether image (and its filename) is for an
            individual plot (True), or for many plots (False) (default: False).
        '''
        if name_long is None:  # finds first '-' after last '_'
            name_long = fname_in[fname_in.find('-',fname_in.rfind('_')):]
        self.fname_in = fname_in
        self.fname_shp = fname_shp
        self.base_dir = os.path.split(fname_in)[0]
        self.name_long = name_long
        base_name = os.path.basename(self.fname_in)
        name_short = base_name[:base_name.find(self.name_long)]
        self.name_short = name_short
        self.plot_name = plot_name

        if name_short[-1] == '_' or name_short[-1] == '-':
            name_short = name_short[:-1]
        if plot_name is True:
#            self.name_plot = 'plot' + name_short[name_short.rfind('_'):]
            self.name_plot = 'plot' + name_short[name_short.find('_'):]
            plot = name_short[name_short.rfind('_')+1:]
            self.plot = plot
        else:
            self.name_plot = None
            self.plot = None

#        self._read_envi_gdal()
        self._read_envi_hdr()
        self._read_envi_spy()

    def spectral_clip_and_smooth_batch(self, base_dir, name_long=None,
                                       name_append=None,
                                       wl_bands=[[0, 420], [760, 776],
                                                 [813, 827], [880, 1000]],
                                       spectra_smooth=True, window_size=11,
                                       order=2, save_out=True,
                                       interleave='bip', level=0):
        '''
        Does a batch clip and smooth for all files in a directory with the
        given file extension. Saves some descriptive statistics that result
        from the processing.

        Parameters:
            base_dir (`str`): Directory to search for images to process
                default (None)
            name_long (str): Spectronon processing appends processing names to
                the filenames; this indicates those processing names that are
                repetitive and can be deleted from the filename following
                processing.
            name_append (`str`): text to append to end of filename; used to
                describe this particular manipulation (default:
                'smooth_spec_clip').
            wl_bands (`list`): minimum and maximum wavelenths to clip from
                image; if multiple groups of wavelengths should be cut, this
                should be a list of lists. For example, wl_bands=[760, 776]
                will clip all bands greater than 760.0 nm and less than 776.0
                nm; wl_bands = [[0, 420], [760, 776], [813, 827], [880, 1000]]
                will clip all band less than 420.0 nm, bands greater than 760.0
                nm and less than 776.0 nm, bands greater than 813.0 nm and less
                than 827.0 nm, and bands greater than 880 nm (default).
            spectra_smooth (`bool`): Indicates whether Savitzky-Golay smoothing
                should be performed on the spectral domain (default: True).
            window_size (`int`): the length of the window; must be an odd integer
                number (default: 11).
            order (`int`): the order of the polynomial used in the filtering; must
                be less than `window_size` - 1 (default: 2).
            save_out (`bool): indicates whether manipulated image should be
                saved to disk (defaul: True).
            interleave (`str`): interleave (and file extension) of input and
                manipulated file; (default: 'bip').
            level (`int`): Number of levels to search in the directory
                (default: 0)
        '''
        fname_list = self._recurs_dir(base_dir=base_dir,
                                      search_exp='.' + interleave, level=0)
        self.base_dir = base_dir
        base_dir_out, name_print = self._save_file_setup(
                base_dir_out=None, folder_name='smooth_spec_clip')
        df_smooth_stats = pd.DataFrame(columns=['fname', 'mean', 'std', 'cv'])
#        name_long = ('-Radiance From Raw Data-Georectify Airborne Datacube-'
#                     'Reflectance from Radiance Data and Measured Reference Spectrum')

        for idx, fname_in in enumerate(fname_list):
            if name_long is None:  # finds first '-' after last '_'
                name_long = fname_in[fname_in.find('-',fname_in.rfind('_')):]
            self.read_cube(fname_in, name_long=name_long)
            self.spectral_clip_and_smooth(base_dir_out=base_dir_out,
                                          name_append=name_append,
                                          wl_bands=wl_bands,
                                          spectra_smooth=spectra_smooth,
                                          window_size=window_size,
                                          order=order, save_out=save_out,
                                          interleave=interleave)
            mean = np.nanmean(self.array_smooth)
            std = np.nanstd(self.array_smooth)
            cv = std/mean
            df_smooth_temp = pd.DataFrame([[fname_in, mean, std, cv]],
                                          columns=['fname', 'mean', 'std',
                                                   'cv'])
            df_smooth_stats = df_smooth_stats.append(df_smooth_temp,
                                                     ignore_index=True)
        fname_stats = os.path.join(base_dir_out, 'spec_clip_smooth_stats.csv')
        df_smooth_stats.to_csv(fname_stats)

    def spectral_clip_and_smooth(self, base_dir_out=None, name_append=None,
                                 wl_bands=[[0, 420], [760, 776], [813, 827],
                                           [880, 1000]],
                                 spectra_smooth=True, window_size=11, order=2,
                                 save_out=True, interleave='bip'):
        '''
        Removes/clips designated wavelength bands from image, smooths data on
        the spectral domain (optional), and saves cleaned image to file

        Parameters:
            base_dir_out (`str`): directory path to save file to; if `None`,
                a new folder ("smooth_spec_clip") will be created in the
                directory of the original image file (default: `None`).
            name_append (`str`): text to append to end of filename; used to
                describe this particular manipulation (default:
                'smooth_spec_clip').
            wl_bands (`list`): minimum and maximum wavelenths to clip from
                image; if multiple groups of wavelengths should be cut, this
                should be a list of lists. For example, wl_bands=[760, 776]
                will clip all bands greater than 760.0 nm and less than 776.0
                nm; wl_bands = [[0, 420], [760, 776], [813, 827], [880, 1000]]
                will clip all band less than 420.0 nm, bands greater than 760.0
                nm and less than 776.0 nm, bands greater than 813.0 nm and less
                than 827.0 nm, and bands greater than 880 nm (default).
            spectra_smooth (`bool`): Indicates whether Savitzky-Golay smoothing
                should be performed on the spectral domain (default: True).
            window_size (`int`): the length of the window; must be an odd integer
                number (default: 11).
            order (`int`): the order of the polynomial used in the filtering; must
                be less than `window_size` - 1 (default: 2).
            save_out (`bool): indicates whether manipulated image should be
                saved to disk (defaul: True).
            interleave (`str`): interleave (and file extension) of manipulated
                file; (default: 'bip').
        '''
        base_dir_out, name_print = self._save_file_setup(
                base_dir_out=None, folder_name='smooth_spec_clip')
        if isinstance(wl_bands[0], list):
            spec_clip_groups = [self._get_band_range(grp) for grp in wl_bands]
            spec_clip = list(itertools.chain(*spec_clip_groups))
        else:
            spec_clip = self._get_band_range(wl_bands)

        if name_append is None:
            name_append = 'smooth-spec-clip'
        name_label = (name_print + '-' + str(name_append) + '.' + interleave)
        fname_out_envi = os.path.join(base_dir_out, name_label)
        print('Smoothing and spectrally clipping image: {0}\n'
              ''.format(name_print))

        self.spec_clip = spec_clip
        meta_bands = self.meta_bands.copy()
        for k in self._get_band_num(spec_clip):
            meta_bands.pop(k, None)
        self.meta_bands = meta_bands
        array_clip = np.delete(self.img_sp.asarray(), spec_clip, axis=2)
        if spectra_smooth is True:
            self.array_smooth = self._smooth_image(array_clip,
                                                   window_size=window_size,
                                                   order=2)

            hist_str = (" -> Hyperspectral.spectral_clip_and_smooth[<"
                        "SpecPyFloatText label: 'wl_bands?' value:{0}; "
                        "SpecPyFloatText label: 'window_size?' value:{1}; "
                        "SpecPyFloatText label: 'polynomial_order?' value:{2}>"
                        "]".format(wl_bands, window_size, order))
        else:
            hist_str = (" -> Hyperspectral.spectral_clip_and_smooth[<"
                        "SpecPyFloatText label: 'wl_bands?' value:{0}>"
                        "]".format(wl_bands))
        self.metadata['history'] += hist_str
        self.metadata['bands'] = len(self.meta_bands)
        self.metadata['interleave'] = interleave
        self.metadata['label'] = name_label

        band = []
        wavelength = []
        for idx, (key, val) in enumerate(self.meta_bands.items()):
#            band.append('{0}_{1}'.format(idx, key))
            band.append(idx + 1)
            wavelength.append(val)
#        for b in range(1, len(self.meta_bands)+1):
#            band.append(b)
        band_str = ', '.join(str(b) for b in band)
        band_str = '{' + band_str + '}'
        wavelength.sort()
        wavelength_str = ', '.join(str(wl) for wl in wavelength)
        wavelength_str = '{' + wavelength_str + '}'
        self.metadata['band names'] = band_str
        self.metadata['wavelength'] = wavelength_str

        if save_out is True:
            envi.save_image(fname_out_envi + '.hdr', self.array_smooth,
                            dtype=np.float32, force=True, ext=None,
                            interleave=interleave, metadata=self.metadata)
#            self._write_envi(self.array_smooth, fname_out_envi,
#                             geotransform_out, interleave=interleave,
#                             rewrite_hdr=True)

    def read_spec(self, fname_in=None):
        if fname_in is None:
            fname_in = self.fname_in
        else:
            self.fname_in = fname_in
        self.base_dir = os.path.split(fname_in)[0]

#        self._read_envi_gdal()
        self._read_envi_hdr(fname_in)
        self._read_envi_spy()
        self.array_smooth = self._smooth_image()

    def veg_spectra(self, array_gndvi, thresh=0.55, side='lower'):
        '''
        Gets average spectra across vegetation pixels
        '''
        mask_array = self._mask_array(array_gndvi, thresh=thresh, side=side)

        self.mask_array_3d = np.empty(self.array_smooth.shape)
        for band in range(self.img_sp.nbands):
            self.mask_array_3d[:,:,band] = mask_array.mask
        array_smooth_masked = np.ma.masked_array(self.array_smooth,
                                                 mask=self.mask_array_3d)
        veg_spectra = np.mean(array_smooth_masked, axis=(0, 1))
        return veg_spectra

    def band_math_ratio(self, b1, b2, base_dir_out=None, name=None,
                        save_out=True, interleave='bip'):
        '''
        Calculates a simple ratio spectral index from two input band
        wavelengths

        Parameters:
            b1 (`int` or `float`): the first band to be used in the index; this
                should be the numerator (required).
            b2 (`int` or `float`): the second band to be used in the index;
                this should be the denominator (required).
            base_dir_out (`str`): directory path to save file to; if `None`,
                a new folder ("ratio") will be created in the directory of
                the original image file (default: `None`).
            name (`str`): text to append to end of filename; used to describe
                this particular manipulation (default: 'ndi_b1_b2' where b1 and
                b2 denote band wavelengths used in index).
            save_out (`bool): indicates whether manipulated image should be
                saved to disk (defaul: True).
            interleave (`str`): interleave (and file extension) of manipulated
                file; (default: 'bip').
        '''
        base_dir_out, name_print = self._save_file_setup(
                base_dir_out, folder_name='band_math')
        band1, band2, wl1, wl2 = self._get_band_info_consolidate(b1, b2)
        if name is None:
            name = 'ratio_{0:.0f}_{1:.0f}'.format(wl1, wl2)
        fname_out_envi = os.path.join(
            base_dir_out, (name_print + '_' + str(name) + '.' + interleave))
        print('Calculating normalized difference index for {0}: '
              '{1:.0f}/{2:.0f}'.format(name_print, wl1, wl2))

        if self.array_smooth is not None:
            array = self.array_smooth
        else:
            array = self.img_sp.asarray()
        array_b1 = self._get_band_mean(array, band1)
        array_b2 = self._get_band_mean(array, band2)
        array_index = (array_b1/array_b2)

        geotransform_out = self.geotransform
        if save_out is True:
            self._write_envi(array_index, fname_out_envi, geotransform_out,
                             name, interleave=interleave, modify_hdr=False)
        return array_index

    def band_math_ndi(self, b1=780, b2=559, b3=None, b4=None, b5=None,
                      base_dir_out=None, name_append=None,
                      save_out=True, interleave='bip'):
        '''
        Calculates the spectral index from a list of bands and the "form" of
        the index

        Parameters:
            b1 (`int` or `float`): the first band to be used in the index
                (required).
            b2 (`int` or `float`): the second band to be used in the index
                (required).
            base_dir_out (`str`): directory path to save file to; if `None`,
                a new folder ("band_math") will be created in the directory of
                the original image file (default: `None`).
            name (`str`): text to append to end of filename; used to describe
                this particular manipulation (default: 'ndi_b1_b2' where b1 and
                b2 denote band wavelengths used in index).
            save_out (`bool): indicates whether manipulated image should be
                saved to disk (defaul: True).
            interleave (`str`): interleave (and file extension) of manipulated
                file; (default: 'bip').
        '''
        base_dir_out, name_print = self._save_file_setup(
                base_dir_out, folder_name='band_math')
        band1, band2, wl1, wl2 = self._get_band_info_consolidate(b1, b2)
        if name_append is None:
            name_append = 'ndi_{0:.0f}_{1:.0f}'.format(wl1, wl2)
        name_label = (name_print + '-' + str(name_append) + '.' + interleave)
        fname_out_envi = os.path.join(base_dir_out, name_label)
        print('Calculating normalized difference index for {0}: '
              '({1:.0f}-{2:.0f})/({1:.0f}+{2:.0f})'.format(name_print,
                                                           wl1, wl2))
#        array = self.img_sp.asarray()
        array = self.img_sp.load()
        array_b1 = self._get_band_mean(array, band1)
        array_b2 = self._get_band_mean(array, band2)
        array_index = (array_b1-array_b2)/(array_b1+array_b2)

#        geotransform_out = self.geotransform

        hist_str = (" -> Hyperspectral.band_math_ndi[<"
                    "SpecPyFloatText label: 'b1?' value:{0};"
                    "'b2?' value:{1}>]".format(b1, b2))
        self.metadata['history'] += hist_str
        self.metadata['bands'] = array_index.shape[2]
        self.metadata['interleave'] = interleave
        self.metadata['label'] = name_label

        if save_out is True:
            envi.save_image(fname_out_envi + '.hdr', array_index,
                            dtype=np.float32, force=True, ext=None,
                            interleave=interleave, metadata=self.metadata)

#        if save_out is True:
#            self._write_envi(array_index, fname_out_envi, geotransform_out,
#                             name, interleave=interleave, modify_hdr=False)
        return array_index

    def crop_many(self, base_dir_crop=None):
        '''
        Iterates through all plots, crops each, and saves to file
        '''
        if base_dir_crop is None:
            base_dir_crop = os.path.join(self.base_dir, 'crop')
        if not os.path.isdir(base_dir_crop):
            os.mkdir(base_dir_crop)
#        img_crop = self.ds_in.ReadAsArray(xoff, yoff, xsize, ysize)
        for idx, row in self.df_plots.iterrows():
            plot_id = row['plot_id']
            col_pix = row['col_pix']
            row_pix = row['row_pix']

#        col_pix = df_plots[df_plots['plot_id'] == 2025]['col_pix'].item()
#        row_pix = abs(df_plots[df_plots['plot_id'] == 2025]['row_pix'].item())
            array_img_crop = self.img_ds.ReadAsArray(
                    xoff=abs(col_pix), yoff=abs(row_pix),
                    xsize=abs(self.plot_x_pix - (self.buf_x_pix * 2)),
                    ysize=abs(self.plot_y_pix - (self.buf_y_pix * 2)))


            array_img_crop = self.img_sp.read_subregion(
                    (abs(row_pix), abs(row_pix) + abs(self.plot_y_pix - (self.buf_y_pix * 2))),
                    (abs(col_pix), abs(col_pix) + abs(self.plot_x_pix - (self.buf_x_pix * 2))))
            base_name = os.path.basename(self.fname_in)
            base_name_short = base_name[:base_name.find('gige_') + 7]  # limit of 2 digits in image number (i.e., max of 99)
            if base_name_short[-1] == '_' or base_name_short[-1] == '-':
                base_name_short = base_name_short[:-1]
            fname_out_envi = os.path.join(
                    base_dir_crop, (base_name_short + '_' + str(plot_id) +
                                    '.bsq'))
            print('Cropping plot {0}'.format(plot_id))
            utm_x = self.geotransform[0]
            utm_y = self.geotransform[3]
#            print(plot_id)
#            print(self.df_shp[self.df_shp['plot_id'] == plot_id]['ul_x_utm'])
            if self.fname_shp is not None:
                ul_x_utm = (self.df_shp[self.df_shp['plot_id'] == plot_id]
                            ['ul_x_utm'].item() + self.buf_x_m)
                ul_y_utm = (self.df_shp[self.df_shp['plot_id'] == plot_id]
                            ['ul_y_utm'].item() - self.buf_y_m)
            else:
                ul_x_utm, ul_y_utm = self._get_UTM(col_pix, row_pix, utm_x,
                                                   utm_y, size_x=self.size_x_m,
                                                   size_y=self.size_y_m)
            geotransform_out = [ul_x_utm, self.size_x_m, 0.0, ul_y_utm, 0.0,
                                self.size_y_m]
            self._write_envi(array_img_crop, fname_out_envi, geotransform_out)
            self._modify_hdr(fname_out_envi)
            fname_out_tif = os.path.splitext(fname_out_envi)[0] + '.tif'
            self._write_tif(array_img_crop, fname_out_tif, geotransform_out)

#    def _read_envi_hdr(fname_hdr):
#        '''
#        Reads ENVI .hdr file and
#        '''
#        with open(fname_hdr, 'r') as f:
#            data = f.readlines()
#        matches = []
#        regex1 = re.compile(r'^(.+?)\s*=\s*({\s*.*?\n*.*?})$',re.M|re.I)
#        regex2 = re.compile(r'^(.+?)\s*=\s*(.*?)$',re.M|re.I)
#        for line in data:
#            matches.extend(regex1.findall(line))
#            subhdr = regex1.sub('', line)  # remove from line
#            matches.extend(regex2.findall(subhdr))
#        return dict(matches)
#
    def crop_single_batch(self, fname_sheet, plot_x_pix=90, plot_y_pix=120,
                          interleave='bip', name_append='crop',
                          base_dir_out=None, plot_name=True):
        '''
        Iterates through spreadsheet that provides necessary information about
        how each image should be cropped and how it should be saved
        '''
        df_plots = pd.read_csv(fname_sheet)

        for idx, row in df_plots.iterrows():
            directory = row['directory']
            name_short = row['name_short']
            name_long = row['name_long']
            ext = row['ext']
            pix_e_ul = row['easting_pix']
            pix_n_ul = row['northing_pix']

            fname_in = os.path.join(directory, name_short+name_long+ext)
            self.crop_single(fname_in, pix_e_ul, pix_n_ul,
                             plot_x_pix=plot_x_pix, plot_y_pix=plot_y_pix,
                             interleave=interleave, name_long=name_long,
                             name_append=name_append,
                             base_dir_out=base_dir_out, plot_name=plot_name)

    def crop_single(self, fname_in, pix_e_ul, pix_n_ul, plot_x_pix=90,
                    plot_y_pix=120, interleave='bip',
                    name_long='-Unit Conversion Utility', base_dir_out=None,
                    name_append='crop', plot_name=True):
        '''
        Crops and saves an image

        Parameters:
            fname_in (`str`): hyperspectral image filename to be cropped
            pix_e_ul (`int`): upper left column (easting)to begin cropping
            pix_n_ul (`int`): upper left row (northing) to begin cropping
            plot_x_pix (`int`): number of pixels per row in the cropped image
            plot_y_pix (`int`): number of pixels per colum in the cropped image
            interleave (`str`): interleave (and file extension) of cropped
                image; (default: 'bip').
            name_long (str): Spectronon processing appends processing names to
                the filenames; this indicates those processing names that are
                repetitive and can be deleted from the filename following
                processing.
            base_dir_out (`str`): output directory of the cropped image
                (default: `None`)
            name_append (`str`): text to append to end of filename; used to
                describe this particular manipulation (default: 'crop').
            plot_name (bool): Indicates whether image (and its filename) is for
                an individual plot (True), or for many plots (False) (default:
                True).
        '''
        self.read_cube(fname_in, name_long=name_long, plot_name=plot_name)
        base_dir_out, name_print = self._save_file_setup(
                base_dir_out=base_dir_out, folder_name='crop')

        pix_e_new = pix_e_ul + plot_x_pix
        pix_n_new = pix_n_ul + plot_y_pix
        array_img_crop = self.img_sp.read_subregion((pix_n_ul, pix_n_new),
                                                    (pix_e_ul, pix_e_new))

        if name_append is None:
            name_append = 'crop'
        name_label = (name_print + self._xstr(name_append) + '.' +
                      interleave)
        fname_out_envi = os.path.join(base_dir_out, name_label)
        print('Spatially cropping image: {0}'.format(name_print))

#        if self.name_plot is not None:
#            fname = (self.name_plot + self._xstr(name_append) + '.' +
#                     interleave)
#        else:
#            fname = (self.name_short + self._xstr(name_append) + '.' +
#                     interleave)
#        fname_out_envi = os.path.join(base_dir_out, fname)

#        print('Cropping plot {0}'.format(self.plot))

        map_info_set = self.metadata['map info']
        utm_x = self._get_meta_set(map_info_set, 3)
        utm_y = self._get_meta_set(map_info_set, 4)
        ul_x_utm, ul_y_utm = self._get_UTM(pix_e_ul, pix_n_ul, utm_x,
                                           utm_y, size_x=self.size_x_m,
                                           size_y=self.size_y_m)
        map_info_set = self._modify_meta_set(map_info_set, 3, ul_x_utm)
        map_info_set = self._modify_meta_set(map_info_set, 4, ul_y_utm)
        self.metadata['map info'] = map_info_set
        hist_str = (" -> Hyperspectral.crop_single[<"
                    "SpecPyFloatText label: 'pix_e_ul?' value:{0}; "
                    "SpecPyFloatText label: 'pix_n_ul?' value:{1} >]"
                    "".format(pix_e_ul, pix_n_ul))
        self.metadata['history'] += hist_str
        self.metadata['samples'] = array_img_crop.shape[1]
        self.metadata['lines'] = array_img_crop.shape[0]
        self.metadata['label'] = name_label

        envi.save_image(fname_out_envi + '.hdr', array_img_crop,
                        dtype=np.float32, force=True, ext=None,
                        interleave=interleave, metadata=self.metadata)

        fname_out_tif = os.path.splitext(fname_out_envi)[0] + '.tif'

#            rgb_list = [self._get_band(640)[0],
#                        self._get_band(550)[0],
#                        self._get_band(460)[0]]
#            from spectral import save_rgb
#            save_rgb(fname_out_tif, array_img_crop, rgb_list, format='tiff')

        img_ds = self._get_envi_gdal(fname_in=fname_in)
        projection_out = img_ds.GetProjection()
        img_ds = None  # I only want to use GDAL when I have to..

#        drv = gdal.GetDriverByName('ENVI')
#        drv.Register()
#        img_ds = gdal.Open(fname_in, gdalconst.GA_ReadOnly)
#        projection_out = img_ds.GetProjection()
#        img_ds = None
#        drv = None

        geotransform_out = [ul_x_utm, self.size_x_m, 0.0, ul_y_utm, 0.0,
                            self.size_y_m]
        self._write_tif(array_img_crop, fname_out_tif, projection_out,
                        geotransform_out)

    def show_img(self, array_img, band_r=120, band_g=76, band_b=32,
                 inline=True):
        '''
        Displays the RGB bands
        '''
        if inline is True:
            get_ipython().run_line_magic('matplotlib', 'inline')
        else:
            get_ipython().run_line_magic('matplotlib', 'auto')
        if len(array_img.shape) == 2:
            n_bands = 1
            ysize, xsize = array_img.shape
        elif len(array_img.shape) == 3:
            ysize, xsize, n_bands = array_img.shape
        else:
            raise NotImplementedError('Only 2-D and 3-D arrays can be '
                                      'displayed.')
        if n_bands >= 3:
            try:
                plt.imshow(array_img, (band_r, band_g, band_b))
            except ValueError as err:
                plt.imshow(array_img[:,:,[band_r, band_g, band_b]]*3.5)
#            array_img_out = array_img[:, :, [band_r, band_g, band_b]]
#            array_img_out *= 3.5  # Images are very dark without this

        else:
            plt.imshow(array_img)
#            array_img_out = array_img
#            plt.imshow(array_img_out)
        plt.show()
        print('\n')

