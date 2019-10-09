# -*- coding: utf-8 -*-
import ast
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from osgeo import gdal
from osgeo import gdalconst
from osgeo import ogr
import pandas as pd
import re
import spectral.io.envi as envi
import spectral.io.spyfile as spyfile
import sys


class IO_tools(object):
    '''
    Class for reading and writing hyperspectral data files and accessing,
    interpreting, and modifying its assoicated metadata.
    '''
    def __init__(self, base_dir=None, base_dir_out=None):
        self.base_dir = base_dir
        self.base_dir_out = base_dir_out
        self.fname_in = None
        self.fname_shp = None
        self.img_ds = None
        self.img_sp = None
        self.long_name = None
        self.meta_bands = None
        self.metadata = None
        self.name_long = None
        self.name_plot = None
        self.name_short = None
        self.plot = None

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

    def _del_meta_item(self, my_dict, key):
        '''
        Deletes metadata item and returns new dictionary
        '''
        try:
            del my_dict[key]
        except KeyError:
            print('{0} not a valid key in input dictionary.'.format(key))
        return my_dict

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

    def _modify_meta_set(self, meta_set, idx, value):
        '''
        Modifies meta_set by converting string to list, then adjusting the
        value of an item by its index

        Parameters:
            meta_set (`str`): the string representation of the metadata set
            idx (`int`): index to be modified
            value (`float`, `int`, or `str`): value to replace at idx
        '''
        # idx=None will return the whole set as a list
        meta_set_str = self._get_meta_set(meta_set, idx=None)
        meta_set_str[idx] = str(value)
        set_str = '{'
        for i, item in enumerate(meta_set_str):
            set_str += str(item)
            if i+1 == len(meta_set_str):
                set_str += '}'
            else:
                set_str += ', '
        return set_str

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
            for key, val in enumerate(sorted(ast.literal_eval(
                    self.metadata['wavelength']))):
                meta_bands[key+1] = val
        else:
            try:
                band_names = list(ast.literal_eval(
                        self.metadata['band names']))
                wl_names = list(ast.literal_eval(
                        self.metadata['wavelength']))
            except ValueError as e:
                band_names = list(ast.literal_eval(
                        str(self.metadata['band names'])))
                wl_names = list(ast.literal_eval(
                        str(self.metadata['wavelength'])))
            for idx in range(len(band_names)):
                meta_bands[band_names[idx]] = wl_names[idx]
        self.meta_bands = meta_bands

    def _read_envi_spy(self):
        '''
        Reads ENVI file using Spectral Python; a package with streamlined
        features for hyperspectral IO, memory access, classification, and data
        display
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

    def _read_plot_shp(self):
        '''
        Reads shapefile of plot bounds and record upper left (northwest)
        corner of each plot
        '''
        assert self.df_shp is not None, 'Please load a shapefile\n'
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

        if self.name_plot is not None:
            name_print = self.name_plot
        else:
            name_print = self.name_short
        return base_dir_out, name_print

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
        array = array_mean.reshape(1, 1, len(df_mean))
        envi.save_image(fname_out, array, interleave=interleave, dtype=dtype,
                        byteorder=byteorder, metadata=metadata, force=force,
                        ext=None)

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
        self.show_img(array_img_crop, band_r=band_r, band_g=band_g,
                      band_b=band_b)

    def read_cube(self, fname_in, name_long='-Unit Conversion Utility',
                  plot_name=False):
        '''
        Reads in a hyperspectral datacube

        fname_in (str): filename of datacube to be read
        name_long (str): Spectronon processing appends processing names to
            the filenames; this indicates those processing names that are
            repetitive and can be deleted from the filename following
            processing.
        plot_name (bool): Indicates whether image (and its filename) is for an
            individual plot (True), or for many plots (False) (default: False).
        '''
        self.fname_in = fname_in
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

    @classmethod
    def recurs_dir(cls, base_dir, search_exp='.csv', level=None):
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
                out_files_temp = cls.recurs_dir(new_dir, search_exp)
                if out_files_temp:  # if list is not empty
                    out_files.extend(out_files_temp)  # add items
        return sorted(out_files)

    def write_cube(self, hdr_file, cube_spy, dtype=np.float32,
                   force=False, ext=None, interleave='bip', byteorder=None,
                   metadata=None):
        '''
        Wrapper function that accesses the Spectral Python package to save a
        datacube to file.

        Parameters:
            hdr_file (`str`): Header file path (with the '.hdr' extension).
            cube_spy (`SpyFile` object or `numpy.ndarray`): The hyperspectral
                data cube to save. If `numpy.ndarray`, then metadata (`dict`)
                should also be passed.
            dtype (`numpy.dtype` or `str`): The data type with which to store
                the image. For example, to store the image in 16-bit unsigned
                integer format, the argument could be any of numpy.uint16,
                'u2', 'uint16', or 'H' (default=np.float32).
            force (`bool`): If `hdr_file` or its associated image file exist,
                `force=True` will overwrite the files; otherwise, an exception
                will be raised if either file exists (default=False).
            ext (`str`): The extension to use for saving the image file; if not
                specified, a default extension is determined based on the
                `interleave`. For example, if `interleave`='bip', then `ext` is
                set to 'bip' as well. If `ext` is an empty string, the image
                file will have the same name as the .hdr, but without the
                '.hdr' extension.
            interleave (`str`): The band interleave format to use for writing
                the file; `interleave` should be one of 'bil', 'bip', or 'bsq'
                (default='bip').
            byteorder (`int` or `str`): Specifies the byte order (endian-ness)
                of the data as written to disk. For little endian, this value
                should be either 0 or 'little'. For big endian, it should be
                either 1 or 'big'. If not specified, native byte order will be
                used (default=None).
            metadata (`dict`): Metadata to write to the ENVI .hdr file
                describing the hyperspectral data cube being saved. If
                `SpyFile` object is passed to `cube_spy`, `metadata` will
                overwrite any existing metadata stored by the `SpyFile` object
                (default=None).
        '''
        if ext is None:
            ext = '.' + interleave
        if metadata is None and isinstance(cube_spy, spyfile.SpyFile):
            metadata = cube_spy.metadata
        envi.save_image(hdr_file, cube_spy, dtype=dtype, force=force, ext=ext,
                        interleave=interleave, byteorder=byteorder,
                        metadata=metadata)

    def write_spec_spy(self, fname_out, df_mean, df_std, interleave='bip',
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
        array = array_mean.reshape(1, 1, len(df_mean))
        envi.save_image(fname_out, array, interleave=interleave, dtype=dtype,
                        byteorder=byteorder, metadata=metadata, force=force,
                        ext=None)


class HS_tools(object):
    '''
    Some basic tools for retrieving particular bands, the wavelengths they
    represent, and their order in the data array.
    '''
#    def __init__(self):
#        self.fname_in = None
#        self.base_dir = None
#        self.fname_shp = None
#        self.long_name = None
#
#        self.img_ds = None
#        self.img_sp = None
#        self.array_smooth = None
#        self.mask_array_3d = None
#        self.meta_bands = None
#        self.metadata = None
#        self.plot = None
#
#        self.spec_clip = None
#
#        self.name_short = None
#        self.name_long = None
#        self.name_plot = None
#
#        self.df_plots = pd.DataFrame(columns=['plot_id', 'col_plot',
#                                              'row_plot', 'col_pix',
#                                              'row_pix'])
#        self.df_plots_single = pd.DataFrame(
#                columns=['directory', 'name_short', 'name_long', 'easting_pix',
#                         'northing_pix', 'buffer-x', 'buffer-y'])
#        self.df_shp = pd.DataFrame(columns=['plot_id', 'ul_x_utm', 'ul_y_utm'])

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
        key_band = list(self.meta_bands.keys())[sorted(list(
                self.meta_bands.values())).index(val_target)]
        key_wavelength = sorted(list(self.meta_bands.values()))[key_band-1]
        return key_band, key_wavelength

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

    def _get_band_mean(self, img_array, band_num):
        '''
        Gets the mean value from a list of bands

        Parameters:
            img_array (numpy array): image to evaluate
            band_num (int, list): band number(s) to determine mean value for
        '''
        band_idx = self._get_band_index(band_num)
        if isinstance(band_idx, list):
            array_band = np.mean(img_array[:, :, band_idx], axis=2)
        else:
            array_band = img_array[:, :, band_idx]
        return array_band

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

    def _get_meta_set(self, meta_set, idx=None):
        '''
        Reads a value from metadata "set" (dict-like) based on the index

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
                    if item[0] == ' ':
                        meta_set_str.append(item[1:])
                    else:
                        meta_set_str.append(item)
            else:
                try:
                    meta_set_str.append(float(item))
                except ValueError as e:
                    if item[0] == ' ':
                        meta_set_str.append(item[1:])
                    else:
                        meta_set_str.append(item)
        if idx is None:
            return meta_set_str  # return the whole thing
        else:
            return meta_set_str[idx]

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
                plt.imshow(array_img[:, :, [band_r, band_g, band_b]]*3.5)
#            array_img_out = array_img[:, :, [band_r, band_g, band_b]]
#            array_img_out *= 3.5  # Images are very dark without this

        else:
            plt.imshow(array_img)
#            array_img_out = array_img
#            plt.imshow(array_img_out)
        plt.show()
        print('\n')
