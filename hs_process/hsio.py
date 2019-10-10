# -*- coding: utf-8 -*-
import ast
from matplotlib import pyplot as plt
import numpy as np
import os
from osgeo import gdal
from osgeo import gdalconst
from osgeo import ogr
import pandas as pd
import re
import spectral.io.envi as envi
import spectral.io.spyfile as SpyFile
import sys

class hstools(object):
    '''
    Basic tools for manipulating Spyfiles and accessing their metadata.

    Parameters:
        spyfile (`SpyFile` object): The datacube being accessed and/or
            manipulated.
    '''
    def __init__(self, spyfile=None):
        msg = ('Pleae load a SpyFile (Spectral Python object)')
        assert spyfile is not None, msg

        self.spyfile = spyfile
        self.meta_bands = None

        self._get_meta_bands(spyfile)

    def _get_meta_bands(self, spyfile=None, metadata=None):
        '''
        Retrieves band number and wavelength information from metadata and
        saves as a dictionary

        Parameters:
            metadata (`dict`): dictionary of the metadata
            spyfile (`SpyFile` object or `numpy.ndarray`): The datacube being
                accessed and/or manipulated.
#
#        Returns:
#            meta_bands (`dict`): dictionary of the band information where the
#                band name is the key and wavelength is the value.
        '''
        if spyfile is None:
            spyfile = self.spyfile
        if metadata is None:
            metadata = spyfile.metadata
        meta_bands = {}
        if 'band names' not in metadata.keys():
#            for key, val in enumerate(sorted(ast.literal_eval(
#                    metadata['wavelength']))):
#                meta_bands[key+1] = val

            #aa = metadata['wavelength']
#            print(metadata['wavelength'][1:-1])
#            b = metadata['wavelength'][1:-1].split(', ')
#            print(type(b))
#            for key, val in enumerate(sorted(b)):
#                meta_bands[key+1] = val
            for key, val in enumerate(metadata['wavelength']):
                meta_bands[key+1] = float(val)

        else:
            try:
                band_names = list(ast.literal_eval(metadata['band names']))
                wl_names = list(ast.literal_eval(metadata['wavelength']))
            except ValueError as e:
                band_names = list(ast.literal_eval(
                        str(metadata['band names'])))
                wl_names = list(ast.literal_eval(
                        str(metadata['wavelength'])))
            for idx in range(len(band_names)):
                meta_bands[band_names[idx]] = wl_names[idx]
        self.meta_bands = meta_bands
#        return meta_bands

    def get_band(self, target, spyfile=None):
        '''
        Returns band number of closest target wavelength and that wavelength

        Parameters:
            target (`int` or `float`): the target wavelength to retrive band
                number for (required).
            spyfile (`SpyFile` object): The datacube being accessed and/or
                manipulated; if `None`, uses `hstools.spyfile` (default:
                `None`).

        Example:
            [1] hstools.get_band(703, spyfile)
            >>> (151, 702.52)
        '''
        if spyfile is None:
            spyfile = self.spyfile
        else:
            self.load_spyfile(spyfile)

        val_target = min(list(self.meta_bands.values()),
                         key=lambda x: abs(x-target))
        key_band = list(self.meta_bands.keys())[sorted(list(
                self.meta_bands.values())).index(val_target)]
        key_wavelength = sorted(list(self.meta_bands.values()))[key_band-1]
        return key_band, key_wavelength

    def get_bands(self, band_list, spyfile=None):
        '''
        Gets band numbers and wavelength information for all bands in
        `band_list`.

        Parameters:
            band_list (`list`): the list of bands to get information for
                (required).
            spyfile (`SpyFile` object): The datacube being accessed and/or
                manipulated; if `None`, uses `hstools.spyfile` (default:
                `None`).
        '''
        msg = ('"band_list" must be a list.')
        assert isinstance(band_list, list), msg

        if spyfile is None:
            spyfile = self.spyfile
        else:
            self.load_spyfile(spyfile)

        bands = []
        wls = []
        for band in band_list:
            band_i, wl_i = self._get_band(band)
            bands.append(band_i)
            wls.append(wl_i)
        wls = np.mean(wls)
        return bands, wls

    def get_band_index(self, band_num):
        '''
        Subtracts 1 from `band_num` and returns the band index(es).

        Parameters:
            band_num (`int` or `list`): the target band number(s) to retrive
            the band index for (required).
        '''
        if isinstance(band_num, list):
            band_num = np.array(band_num)
            band_idx = list(band_num - 1)
        else:
            band_idx = band_num - 1
        return band_idx

    def get_band_mean(self, band_list, spyfile=None):
        '''
        Gets the spectral mean of a datacube from a list of bands

        Parameters:
            band_list (`list`): the list of bands to calculate the spectral
                mean for on the datacube (required).
            spyfile (`SpyFile` object or `numpy.ndarray`): The datacube being
                accessed and/or manipulated; if `None`, uses `hstools.spyfile`
                (default: `None`).
        '''
        msg = ('"band_list" must be a list.')
        assert isinstance(band_list, list), msg

        if spyfile is None:
            spyfile = self.spyfile
            array = self.spyfile.load()
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = self.spyfile.load()
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()

        band_idx = self.get_band_index(band_list)
        array_mean = np.mean(array[:, :, band_idx], axis=2)
        return array_mean

    def get_band_num(self, band_idx):
        '''
        Adds 1 to `band_idx` and returns the band number(s).

        Parameters:
            band_idx (`int` or `list`): the target band index(es) to retrive
                the band number for (required).
        '''
        if isinstance(band_idx, list):
            band_idx = np.array(band_idx)
            band_num = list(band_idx + 1)
        else:
            band_num = band_idx + 1
        return band_num

    def get_band_range(self, range_wl, index=True, spyfile=None):
        '''
        Retrieves the band index or band number for all bands within a
        wavelength range.

        Parameters:
            range_wl (`list`): the minimum and maximum wavelength to consider;
                values should be `int` or `float`.
            index (bool): Indicates whether to return the band number (min=1)
                or to return index number (min=0) (default: True)

        Returns:
            band_list (`list`): a list of all bands (either index or number
                depending on how `index` is set) between a range in
                wavelength values.
        '''
        msg = ('"range_wl" must be a `list` or `tuple`.')
        assert isinstance(range_wl, list) or isinstance(range_wl, tuple), msg
        msg = ('"range_wl" must have exactly two items.')
        assert len(range_wl) == 2, msg

        band_min, wl_min = self.get_band(range_wl[0])  # gets closest band
        band_max, wl_max = self.get_band(range_wl[1])
        if wl_min < range_wl[0]:  # ensures its actually within the range
            band_min += 1
        if wl_max > range_wl[1]:
            band_max -= 1
        if index is True:
            band_min = self.get_band_index(band_min)
            band_max = self.get_band_index(band_max)
        band_list = [x for x in range(band_min, band_max+1)]
        return band_list

    def load_spyfile(self, spyfile):
        '''
        Loads a `SpyFile` (Spectral Python object) for data access and/or
        manipulation by the `hstools` class.

        Parameters:
            spyfile (`SpyFile` object): The datacube being accessed and/or
                manipulated.
        '''
        self.spyfile = spyfile
        self._get_meta_bands(spyfile)



class hsio(object):
    '''
    Class for reading and writing hyperspectral data files and accessing,
    interpreting, and modifying its assoicated metadata.

    TODO: Create a temporary Spyfile using envi.create_imamge() and saving to a
        temporary location. This can be used to hold intermediate SpyFiles
        without actually saving them to disk.. (good idea?)
    '''
    def __init__(self, fname_in=None, name_long=None, name_plot=None,
                 name_short=None, str_plot='plot_', individual_plot=False,
                 fname_hdr_spec=None):
        self.fname_in = fname_in
        self.name_long = name_long
        self.name_plot = name_plot
        self.name_short = name_short
        self.str_plot = str_plot
        self.individual_plot = individual_plot
        self.fname_hdr_spec = fname_hdr_spec

        self.base_dir = None
        self.base_dir_spec = None
        self.base_name = None
        self.img_sp = None
        self.spec_sp = None
        self.meta_bands = None

        if fname_in is not None:
            self.fname_hdr = fname_in + '.hdr'
            self.read_cube(fname_hdr=self.fname_hdr, name_long=self.name_long,
                           name_plot=self.name_plot,
                           name_short=self.name_short,
                           individual_plot=individual_plot, overwrite=False)

        if self.fname_hdr_spec is not None:
            self.read_spec(self.fname_hdr_spec)

        self.defaults = defaults

    def _append_hdr_fname(self, fname_hdr, key, value):
        '''
        Appends key and value to ENVI .hdr
        '''
        metadata = self._read_hdr_fname(fname_hdr)
        metadata[key] = value
        self._write_hdr_fname(fname_hdr, metadata)
#        text_append = '{0} = {1}\n'.format(key, value)
#        if os.path.isfile(self.fname_in + '.hdr'):
#            fname_hdr = self.fname_in + '.hdr'
#        else:
#            fname_hdr = self.fname_in[:-4] + '.hdr'
#        with open(fname_hdr, 'a') as f:
#            f.write(text_append)

    def _del_meta_item(self, key, metadata=None):
        '''
        Deletes metadata item from SpyFile object.

        Parametes:
            metadata (`dict`): dictionary of the metadata

        Returns:
            metadata (`dict`): Dictionary containing the modified metadata.
        '''
        if metadata is None:
            metadata = self.img_sp.metadata
        try:
            del metadata[key]
        except KeyError:
            print('{0} not a valid key in input dictionary.'.format(key))
        self.img_sp.metadata = metadata
        return metadata


    def _get_meta_bands(self, metadata=None, spec=False):
        '''
        Retrieves band number and wavelength information from metadata and
        saves as a dictionary

        Parameters:
            metadata (`dict`): dictionary of the metadata

        Returns:
            meta_bands (`dict`): dictionary of the band information where the
                band name is the key and wavelength is the value.
            spec (`bool`): Whether the file to be read is an image (`False`) or
                a spectrum (`True`; default: `False`).
        '''
        if spec is False:
            spyfile = self.img_sp
        else:
            spyfile = self.spec_sp
        if metadata is None:
            metadata = spyfile.metadata
        meta_bands = {}
        if 'band names' not in metadata.keys():
#            for key, val in enumerate(sorted(ast.literal_eval(
#                    metadata['wavelength']))):
#                meta_bands[key+1] = val

            #aa = metadata['wavelength']
#            print(metadata['wavelength'][1:-1])
#            b = metadata['wavelength'][1:-1].split(', ')
#            print(type(b))
#            for key, val in enumerate(sorted(b)):
#                meta_bands[key+1] = val
            for key, val in enumerate(metadata['wavelength']):
                meta_bands[key+1] = float(val)

        else:
            try:
                band_names = list(ast.literal_eval(metadata['band names']))
                wl_names = list(ast.literal_eval(metadata['wavelength']))
            except ValueError as e:
                band_names = list(ast.literal_eval(
                        str(metadata['band names'])))
                wl_names = list(ast.literal_eval(
                        str(metadata['wavelength'])))
            for idx in range(len(band_names)):
                meta_bands[band_names[idx]] = wl_names[idx]
        self.meta_bands = meta_bands

    def _get_meta_set(self, meta_set, idx=None):
        '''
        Reads metadata "set" (i.e., string representation of a Python set;
        common in .hdr files), taking care to remove leading and trailing
        spaces.

        Parameters:
            meta_set (`str`): the string representation of the metadata set
            idx (`int`): index to be read; if `None`, the whole list is
                returned (default: `None`).

        Returns:
            metadata_list (`list` or `str`): List of metadata set items (as
                `str`), or if idx is not `None`, the item in the position
                described by `idx`.
        '''
        meta_set_list = meta_set[1:-1].split(",")
        metadata_list = []
        for item in meta_set_list:
            if str(item)[::-1].find('.') == -1:
                try:
                    metadata_list.append(int(item))
                except ValueError as e:
                    if item[0] == ' ':
                        metadata_list.append(item[1:])
                    else:
                        metadata_list.append(item)
            else:
                try:
                    metadata_list.append(float(item))
                except ValueError as e:
                    if item[0] == ' ':
                        metadata_list.append(item[1:])
                    else:
                        metadata_list.append(item)
        if idx is None:
            return metadata_list  # return the whole thing
        else:
            return metadata_list[idx]

    def _modify_meta_set(self, meta_set, idx, value):
        '''
        Modifies metadata "set" (i.e., string representation of a Python set;
        common in .hdr files) by converting string to list, then adjusts the
        value of an item by its index.

        Parameters:
            meta_set (`str`): the string representation of the metadata set
            idx (`int`): index to be modified; if `None`, the whole meta_set is
                returned (default: `None`).
            value (`float`, `int`, or `str`): value to replace at idx

        Returns:
            set_str (`str`):
        '''
        metadata_list = self._get_meta_set(meta_set, idx=None)
        metadata_list[idx] = str(value)
        set_str = '{' + ', '.join(str(x) for x in metadata_list) + '}'
        return set_str
#        set_str = '{'
#        for i, item in enumerate(metadata_list):
#            set_str += str(item)
#            if i+1 == len(metadata_list):
#                set_str += '}'
#            else:
#                set_str += ', '
#        return set_str

    def _parse_fname(self, fname_hdr=None, str_plot='plot_', overwrite=True):
        '''
        Parses the filename for `name_long` (text after the first dash,
        inclusive), `name_short` (text before the first dash), and `name_plot`
        (numeric text following `str_plot`).

        Parameters:
            fname_hdr (`str`): input filename.
            str_plot (`str`): text to search for that precedes the numeric text
                that describes the plot number.
            overwrite (`bool`): whether the class instances of `name_long`,
                `name_short`, and `name_plot` should be overwritten based on
                `fname_in` (default: `True`).
        '''
        if fname_hdr is None:
            fname_hdr = self.fname_in + '.hdr'
        if os.path.splitext(fname_hdr)[1] == '.hdr':  # modify self.fname_in based on new file
            fname_in = os.path.splitext(fname_hdr)[0]
        else:
            fname_hdr = fname_hdr + '.hdr'
            fname_in = os.path.splitext(fname_hdr)[0]
        self.fname_in = fname_in
        self.fname_hdr = fname_hdr

        self.base_dir = os.path.dirname(fname_in)
        base_name = os.path.basename(fname_in)
        self.base_name = base_name
        if overwrite is True:
            self.name_long = base_name[base_name.find(
                    '-', base_name.rfind('_')):]
            self.name_short = base_name[:base_name.find(
                    '-', base_name.rfind('_'))]
            s = base_name
            name_plot = s[s.find(str_plot) + len(str_plot):s.find('_pika')]
            if len(name_plot) > 8:  # then it must have gone wrong
                name_plot = self.name_short[self.name_short.rfind('_')+1:]
            try:
                int(name_plot)
            except ValueError:  # give up..
                name_plot = None
            self.name_plot = name_plot
        else:
            if self.name_long is None:  # finds first '-' after last '_'
                self.name_long = base_name[base_name.find(
                        '-', base_name.rfind('_')):]
            if self.name_short is None:
                self.name_short = base_name[:base_name.find(
                        '-', base_name.rfind('_'))]
            if self.name_plot is None:
                s = base_name
                name_plot = s[s.find(str_plot) + len(str_plot):s.find('_pika')]
                if len(name_plot) > 8:  # then it must have gone wrong
                    name_plot = self.name_short[self.name_short.rfind('_')+1:]
                try:
                    int(name_plot)
                except ValueError:  # give up..
                    name_plot = None
                self.name_plot = name_plot

    def _read_envi(self, spec=False):
        '''
        Reads ENVI file using Spectral Python; a package with streamlined
        features for hyperspectral IO, memory access, classification, and data
        display

        Parameters:
            spec (`bool`): Whether the file to be read is an image (`False`) or
                a spectrum (`True`; default: `False`).
        '''
        if spec is False:
#            fname_hdr = self.fname_in + '.hdr'
            fname_hdr = self.fname_hdr
            try:
                self.img_sp = envi.open(fname_hdr)
            except envi.MissingEnviHeaderParameter as e:  # Resonon excludes
                err = str(e)
                key = err[err.find('"') + 1:err.rfind('"')]
                if key == 'byte order':
                    self._append_hdr_fname(fname_hdr, key, 0)
                else:
                    print(err)
                self.img_sp = envi.open(fname_hdr)
        else:
            fname_hdr_spec = self.fname_hdr_spec
            try:
                self.spec_sp = envi.open(fname_hdr_spec)
            except envi.MissingEnviHeaderParameter as e:  # Resonon excludes
                err = str(e)
                key = err[err.find('"') + 1:err.rfind('"')]
                if key == 'byte order':
                    self._append_hdr_fname(fname_hdr, key, 0)
                else:
                    print(err)
                self.spec_sp = envi.open(fname_hdr_spec)
#        self._get_meta_bands(spec=spec)
        tools = hstools(self.img_sp)
        self.meta_bands = tools.meta_bands

    def _read_envi_gdal(self, fname_in=None):
        '''
        Reads and ENVI file via GDAL

        Parameters:
            fname_in (`str`): filename of the ENVI file to read (not the .hdr;
                default: `None`).

        Returns:
            img_ds (`GDAL object`): GDAL dataset containing the image
            infromation
        '''
        if fname_in is None:
            fname_in = self.fname_in
        drv = gdal.GetDriverByName('ENVI')
        drv.Register()
        img_ds = gdal.Open(fname_in, gdalconst.GA_ReadOnly)
        if img_ds is None:
            sys.exit("Image not loaded. Check file path and try again.")
        return img_ds

    def _read_hdr_fname(self, fname_hdr=None):
        '''
        Reads the .hdr file and returns a dictionary of the metadata

        Parameters:
            fname_hdr (`str`): filename of .hdr file

        Returns:
            metadata (`dict`): dictionary of the metadata
        '''
        if fname_hdr is None:
            fname_hdr = self.fname_in + '.hdr'
        if not os.path.isfile(fname_hdr):
            fname_hdr = self.fname_in
        assert os.path.isfile(fname_hdr), 'Could not find .hdr file.'
        with open(fname_hdr, 'r') as f:
            data = f.readlines()
        matches = []
        regex1 = re.compile(r'^(.+?)\s*=\s*({\s*.*?\n*.*?})$', re.M | re.I)
        regex2 = re.compile(r'^(.+?)\s*=\s*(.*?)$', re.M | re.I)
        for line in data:
            matches.extend(regex1.findall(line))
            subhdr = regex1.sub('', line)  # remove from line
            matches.extend(regex2.findall(subhdr))
        metadata = dict(matches)
        return metadata

    def _write_hdr_fname(self, fname_hdr=None, metadata=None):
        '''
        Writes/overwrites an ENVI .hdr file from the beginning using a metadata
        dictionary.

        Parameters:
            fname_hdr (`str`): filename of .hdr file to write (default:
                `None`).
            metadata (`dict`): dictionary of the metadata (default: `None`).
        '''
        if fname_hdr is None:
            fname_hdr = self.fname_in + '.hdr'
        if metadata is None:
            metadata = self.img_sp.metadata
        _, ext = os.path.splitext(fname_hdr)
        if ext != '.hdr':
            fname_hdr = fname_hdr + '.hdr'

        with open(fname_hdr, 'w') as f:
            f.write('ENVI\n')
            for key, val in sorted(metadata.items()):
                f.write('{0} = {1}\n'.format(key, val))
            f.write('\n')

    def _xstr(self, s):
        '''
        Function for safely returning an empty string (e.g., `None`).

        Parameters:
            s (`str` or `None`): the variable that may contain a string.
        '''
        if s is None:
            return ''
        return str('-' + s)

    def read_cube(self, fname_hdr=None, name_long=None, name_plot=None,
                  name_short=None, individual_plot=False, overwrite=True):
        '''
        Reads in a hyperspectral datacube

        Parameters:
            fname_hdr (str): filename of datacube to be read (default: `None`).
            name_long (str): Spectronon processing appends processing names to
                the filenames; this indicates those processing names that are
                repetitive and can be deleted from the filename following
                processing (default: `None`).
            name_plot (`str`): numeric text that describes the plot number
                (default: `None`).
            name_short (`str`): The base name of the image file (see note above
                about `name_long`; default: `None`).
            individual_plot (`bool`): Indicates whether image (and its
                filename) is for an individual plot (`True`), or for many plots
                (`False`; default: `False`).
            overwrite (`bool`): Whether to overwrite any of the previous
                user-passed variables, including `name_long`, `name_plot`, and
                `name_short`; any of the current user-passed variables will
                overwrite previous ones whether `overwrite` is `True` or
                `False` (default: `False`).
        '''
        # Basically resets static __init__ variables for the new filename
        # If variables are already set and overwrite is False, they will remain
        # the same; if variables are set and overwrite is True, they will be
        # overwritten
        self._parse_fname(fname_hdr, self.str_plot, overwrite=overwrite)

        # The following ensures that user-passed variables have priority
        if not os.path.isfile(fname_hdr):
            fname_hdr = self.fname_in
        if name_long is not None:
            self.name_long = name_long
        if name_plot is not None:
            self.name_plot = name_plot
        if name_short is not None:
            self.name_short = name_short

        if self.name_short[-1] == '_' or self.name_short[-1] == '-':
            self.name_short = self.name_short[:-1]
        if individual_plot is True and name_plot is None:
            name_plot = name_short[name_short.rfind('_')+1:]

        self.individual_plot = individual_plot
        self._read_envi()

    def read_spec(self, fname_hdr_spec):
        '''
        Reads in a hyperspectral spectrum file

        Parameters:
            fname_hdr_spec (`str`): filename of spectra to be read.
        '''
        assert os.path.isfile(fname_hdr_spec), 'Could not find .hdr file.'
        self.fname_hdr_spec = fname_hdr_spec
        self.base_dir_spec = os.path.dirname(fname_hdr_spec)
        self._read_envi(spec=True)

    def set_io_defaults(self, dtype=False, force=None, ext=False,
                        interleave=False, byteorder=False):
        '''
        Sets any of the ENVI file writing parameters to `hsio`; if any
        parameter is left unchanged from its default, it will remain as-is
        (it will not be set).

        Parameters:
            dtype (`numpy.dtype` or `str`): The data type with which to store
                the image. For example, to store the image in 16-bit unsigned
                integer format, the argument could be any of numpy.uint16,
                'u2', 'uint16', or 'H' (default=`False`).
            force (`bool`): If `hdr_file` or its associated image file exist,
                `force=True` will overwrite the files; otherwise, an exception
                will be raised if either file exists (default=`None`).
            ext (`str`): The extension to use for saving the image file; if not
                specified, a default extension is determined based on the
                `interleave`. For example, if `interleave`='bip', then `ext` is
                set to 'bip' as well. If `ext` is an empty string, the image
                file will have the same name as the .hdr, but without the
                '.hdr' extension (default: `False`).
            interleave (`str`): The band interleave format to use for writing
                the file; `interleave` should be one of 'bil', 'bip', or 'bsq'
                (default=`False`).
            byteorder (`int` or `str`): Specifies the byte order (endian-ness)
                of the data as written to disk. For little endian, this value
                should be either 0 or 'little'. For big endian, it should be
                either 1 or 'big'. If not specified, native byte order will be
                used (default=`False`).
        '''
        if dtype is not False:
            self.defaults.dtype = dtype
        if force is not None:
            self.defaults.force = force
        if ext is not False:
            self.defaults.ext = ext
        if interleave is not False:
            self.defaults.interleave = interleave
        if byteorder is not False:
            self.defaults.byteorder = byteorder

    def show_img(self, spyfile=None, band_r=120, band_g=76, band_b=32,
                 inline=True):
        '''
        Displays a datacube as a 3-band RGB image.

        Parameters:
            spyfile (`SpyFile` object or `numpy.ndarray`): The data cube to
                display; if `None`, loads from `self.img_sp` (default:
                `None`).
            band_r (`int`): Band to display on the red channel (default: 120)
            band_g (`int`): Band to display on the green channel (default: 76)
            band_b (`int`): Band to display on the blue channel (default: 32)
            inline (`bool`): If `True`, displays in the IPython console; else
                displays in a pop-out window (default: `True`).
        '''
        if inline is True:
            get_ipython().run_line_magic('matplotlib', 'inline')
        else:
            get_ipython().run_line_magic('matplotlib', 'auto')

        if spyfile is None:
            spyfile = self.img_sp
        if isinstance(spyfile, SpyFile.SpyFile):
            array = spyfile.load()
        else:
            assert isinstance(spyfile, np.ndarray)
            array = spyfile

        if len(array.shape) == 2:
            n_bands = 1
            ysize, xsize = array.shape
        elif len(array.shape) == 3:
            ysize, xsize, n_bands = array.shape
        else:
            raise NotImplementedError('Only 2-D and 3-D arrays can be '
                                      'displayed.')
        if n_bands >= 3:
            try:
                plt.imshow(array, (band_r, band_g, band_b))
            except ValueError as err:
                plt.imshow(array[:, :, [band_r, band_g, band_b]]*5.0)
#            array_img_out = array_img[:, :, [band_r, band_g, band_b]]
#            array_img_out *= 3.5  # Images are very dark without this

        else:
            plt.imshow(array)
#            array_img_out = array_img
#            plt.imshow(array_img_out)
        plt.show()
        print('\n')

    def write_cube(self, hdr_file, spyfile, dtype=np.float32,
                   force=False, ext=None, interleave='bip', byteorder=None,
                   metadata=None):
        '''
        Wrapper function that accesses the Spectral Python package to save a
        datacube to file.

        Parameters:
            hdr_file (`str`): Output header file path (with the '.hdr'
                extension).
            spyfile (`SpyFile` object or `numpy.ndarray`): The hyperspectral
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
                '.hdr' extension (default: None).
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
                `SpyFile` object is passed to `spyfile`, `metadata` will
                overwrite any existing metadata stored by the `SpyFile` object
                (default=None).
        '''
        if ext is None:
            ext = '.' + interleave
        if metadata is None and isinstance(spyfile, SpyFile.SpyFile):
            metadata = spyfile.metadata
        if os.path.splitext(hdr_file)[1] != '.hdr':
            hdr_file = hdr_file + '.hdr'

        envi.save_image(hdr_file, spyfile, dtype=dtype, force=force, ext=ext,
                        interleave=interleave, byteorder=byteorder,
                        metadata=metadata)

    def write_spec(self, hdr_file, df_mean, df_std, dtype=np.float32,
                   force=True, ext='.spec', interleave='bip', byteorder=0,
                   metadata=None):
        '''
        Wrapper function that accesses the Spectral Python package to save a
        single spectra to file.

        Parameters:
            hdr_file (`str`): Output header file path (with the '.hdr'
                extension).
            df_mean (`pandas.DataFrame`): Mean spectra, stored as a df row,
                where columns are the bands.
            df_std (`pandas.DataFrame`): Standard deviation of each spectra,
                stored as a df row, where columns are the bands. This will be
                saved to the .hdr file.
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
                '.hdr' extension (default: '.spec').
            interleave (`str`): The band interleave format to use for writing
                the file; `interleave` should be one of 'bil', 'bip', or 'bsq'
                (default='bip').
            byteorder (`int` or `str`): Specifies the byte order (endian-ness)
                of the data as written to disk. For little endian, this value
                should be either 0 or 'little'. For big endian, it should be
                either 1 or 'big'. If not specified, native byte order will be
                used (default=None).
            metadata (`dict`): Metadata to write to the ENVI .hdr file
                describing the spectra being saved; if `None`, will try to pull
                metadata template from self.img_sp.metadata (default=None).
        '''
        if ext is None:
            ext = '.' + interleave
        if metadata is None:
            metadata = self.img_sp.metadata
        if os.path.splitext(hdr_file)[1] != '.hdr':
            hdr_file = hdr_file + '.hdr'

        metadata = self._del_meta_item(metadata, 'map info')
        metadata = self._del_meta_item(metadata, 'history')
        metadata = self._del_meta_item(metadata, 'original cube file')
        metadata = self._del_meta_item(metadata, 'pointlist')
        metadata = self._del_meta_item(metadata, 'boundary')
        metadata = self._del_meta_item(metadata, 'label')

        metadata['band names'] = '{' + ', '.join(str(e) for e in list(
                self.meta_bands.keys())) + '}'
        std = df_std.to_dict()
        metadata['stdev'] = '{' + ', '.join(str(e) for e in list(
                std.values())) + '}'

        array_mean = df_mean.to_numpy()
        array = array_mean.reshape(1, 1, len(df_mean))
        envi.save_image(hdr_file, array, interleave=interleave, dtype=dtype,
                        byteorder=byteorder, metadata=metadata, force=force,
                        ext=ext)

    def write_tif(self, fname_tif, spyfile=None,
                  projection_out=None, geotransform_out=None):
        '''
        Wrapper function that accesses the GDAL Python package to save a
        small datacube subset (i.e., three bands or less) to file.

        Parameters:
            fname_tif (`str`): Output image file path (with the '.tif'
                extension).
            spyfile (`SpyFile` object or `numpy.ndarray`): The data cube to
                save. If `numpy.ndarray`, then metadata (`dict`) should also be
                passed.
            projection_out (`str`): (default: `self.projection_out`)
            geotransform_out (`str`): (default: `self.geotransform_out`)

        TOOD:
            Use rasterio package instead of GDAL
        '''
        if spyfile is None:
            spyfile = self.img_sp
        if isinstance(spyfile, SpyFile.SpyFile):
            array = spyfile.load()
        else:
            assert isinstance(spyfile, np.ndarray)
            array = spyfile

        if projection_out is None or geotransform_out is None:
            print('Either `projection_out` is `None` or `geotransform_out` is '
                  '`None` (or both are). Retrieving projection and '
                  'geotransform information by loading `self.fname_in` via '
                  'GDAL. Be sure this is appropriate for the data you are '
                  'trying to write\n.')
            img_ds = self._read_envi_gdal()
            projection_out = img_ds.GetProjection()
            geotransform_out = img_ds.GetGeoTransform()

        drv = gdal.GetDriverByName('GTiff')
        drv.Register()
        ysize, xsize, bands = array.shape
        tif_out = drv.Create(fname_tif, xsize, ysize, 3, gdal.GDT_Float32)
        tif_out.SetProjection(projection_out)
        tif_out.SetGeoTransform(geotransform_out)

        band_b = self._get_band(460)[0]
        band_g = self._get_band(550)[0]
        band_r = self._get_band(640)[0]
        band_list = [band_r, band_g, band_b]  # backwards for RGB display

        array_img = None
        for idx, band in enumerate(band_list):
            array_band = array[:, :, band-1]
            if len(array_band.shape) > 2:
                array_band = array_band.reshape((array_band.shape[0],
                                                 array_band.shape[1]))
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
        self.show_img(array, band_r=band_r, band_g=band_g,
                      band_b=band_b)


class defaults:
    '''
    Class containing all defaults for writing an ENVI datacube to file.

    Parameters:
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
            '.hdr' extension (default: None).
        interleave (`str`): The band interleave format to use for writing
            the file; `interleave` should be one of 'bil', 'bip', or 'bsq'
            (default='bip').
        byteorder (`int` or `str`): Specifies the byte order (endian-ness)
            of the data as written to disk. For little endian, this value
            should be either 0 or 'little'. For big endian, it should be
            either 1 or 'big'. If not specified, native byte order will be
            used (default=None).
    '''
    dtype = np.float32
    force = False
    ext = None
    interleave = 'bip'
    byteorder = None



#class IO_tools2(object):
#    '''
#    Class for reading and writing hyperspectral data files and accessing,
#    interpreting, and modifying its assoicated metadata.
#    '''
#    def __init__(self, fname_in, base_dir_out=None):
#        self.base_dir = base_dir
#        self.base_dir_out = base_dir_out
#        self.fname_in = None
#        self.fname_shp = None
#        self.img_ds = None
#        self.img_sp = None
#        self.long_name = None
#        self.meta_bands = None
#        self.metadata = None
#        self.name_long = None
#        self.name_plot = None
#        self.name_short = None
#        self.plot = None

#    def _append_hdr(self, key, value):
#        '''
#        Appends key and value to ENVI .hdr
#        '''
#        text_append = '{0} = {1}\n'.format(key, value)
#        if os.path.isfile(self.fname_in + '.hdr'):
#            fname_hdr = self.fname_in + '.hdr'
#        else:
#            fname_hdr = self.fname_in[:-4] + '.hdr'
#        with open(fname_hdr, 'a') as f:
#            f.write(text_append)
#
#    def _xstr(self, s):
#        if s is None:
#            return ''
#        return str('-' + s)
#
#    def _del_meta_item(self, my_dict, key):
#        '''
#        Deletes metadata item and returns new dictionary
#        '''
#        try:
#            del my_dict[key]
#        except KeyError:
#            print('{0} not a valid key in input dictionary.'.format(key))
#        return my_dict
#
#    def _get_envi_gdal(self, fname_in=None):
#        '''
#        GDAL dataset isn't saved to self, so this function gets and returns the
#        GDAL object if necessary.
#        '''
#        if fname_in is None:
#            fname_in = self.fname_in
#        drv = gdal.GetDriverByName('ENVI')
#        drv.Register()
#        img_ds = gdal.Open(fname_in, gdalconst.GA_ReadOnly)
#        if img_ds is None:
#            sys.exit("Image not loaded. Check file path and try again.")
#        return img_ds
#
#    def _modify_meta_set(self, meta_set, idx, value):
#        '''
#        Modifies meta_set by converting string to list, then adjusting the
#        value of an item by its index
#
#        Parameters:
#            meta_set (`str`): the string representation of the metadata set
#            idx (`int`): index to be modified
#            value (`float`, `int`, or `str`): value to replace at idx
#        '''
#        # idx=None will return the whole set as a list
#        meta_set_str = self._get_meta_set(meta_set, idx=None)
#        meta_set_str[idx] = str(value)
#        set_str = '{'
#        for i, item in enumerate(meta_set_str):
#            set_str += str(item)
#            if i+1 == len(meta_set_str):
#                set_str += '}'
#            else:
#                set_str += ', '
#        return set_str
#
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
#
#    def _read_envi_hdr(self, fname_in=None):
#        '''
#        Reads ENVI .hdr file and
#        '''
#        if fname_in is None:
#            fname_in = self.fname_in
#        if os.path.isfile(fname_in + '.hdr'):
#            fname_hdr = fname_in + '.hdr'
#        else:
#            fname_hdr = fname_in[:-4] + '.hdr'
#        with open(fname_hdr, 'r') as f:
#            data = f.readlines()
#        matches = []
#        regex1 = re.compile(r'^(.+?)\s*=\s*({\s*.*?\n*.*?})$', re.M | re.I)
#        regex2 = re.compile(r'^(.+?)\s*=\s*(.*?)$', re.M | re.I)
#        for line in data:
#            matches.extend(regex1.findall(line))
#            subhdr = regex1.sub('', line)  # remove from line
#            matches.extend(regex2.findall(subhdr))
#        self.metadata = dict(matches)
#
#        meta_bands = {}
#        if 'band names' not in self.metadata.keys():
#            for key, val in enumerate(sorted(ast.literal_eval(
#                    self.metadata['wavelength']))):
#                meta_bands[key+1] = val
#        else:
#            try:
#                band_names = list(ast.literal_eval(
#                        self.metadata['band names']))
#                wl_names = list(ast.literal_eval(
#                        self.metadata['wavelength']))
#            except ValueError as e:
#                band_names = list(ast.literal_eval(
#                        str(self.metadata['band names'])))
#                wl_names = list(ast.literal_eval(
#                        str(self.metadata['wavelength'])))
#            for idx in range(len(band_names)):
#                meta_bands[band_names[idx]] = wl_names[idx]
#        self.meta_bands = meta_bands
#
#    def _read_envi_spy(self):
#        '''
#        Reads ENVI file using Spectral Python; a package with streamlined
#        features for hyperspectral IO, memory access, classification, and data
#        display
#        '''
#        meta = self.metadata
#        if 'byte order' not in meta.keys():
#            meta['byte order'] = 0
#            self._append_hdr('byte order', 0)
#        # Note: img_sp.asarray() is always in .bsq order (x, y, z)
#        self.img_sp = envi.open(self.fname_in + '.hdr')
#
#        try:
#            self.ul_y_m = float(self.img_sp.metadata['map info'][3])
#            self.ul_x_m = float(self.img_sp.metadata['map info'][4])
#            self.size_x_m = float(self.img_sp.metadata['map info'][5])
#            self.size_y_m = float(self.img_sp.metadata['map info'][6])
#        except KeyError as e:
#            self.ul_y_m = None
#            self.ul_x_m = None
#            self.size_x_m = None
#            self.size_y_m = None
#
#    def _read_plot_shp(self):
#        '''
#        Reads shapefile of plot bounds and record upper left (northwest)
#        corner of each plot
#        '''
#        assert self.df_shp is not None, 'Please load a shapefile\n'
#        df_shp = self.df_shp.copy()
#        drv = ogr.GetDriverByName('ESRI Shapefile')
#        ds_shp = drv.Open(self.fname_shp, 0)
#        if ds_shp is None:
#            print('Could not open {0}'.format(self.fname_shp))
#        layer = ds_shp.GetLayer()
#
#        for feat in layer:
#            geom = feat.GetGeometryRef()
#            bounds = geom.GetBoundary()
#            bounds_dict = json.loads(bounds.ExportToJson())
#            bounds_coords = bounds_dict['coordinates']
#            plot_id = feat.GetField('plot')
#            x, y = zip(*bounds_coords)
#            ul_x_utm = min(x)
#            ul_y_utm = max(y)
#            df_temp = pd.DataFrame(data=[[plot_id, ul_x_utm, ul_y_utm]],
#                                   columns=df_shp.columns)
#            df_shp = df_shp.append(df_temp, ignore_index=True)
#            self.df_shp = df_shp
#
#    def _rewrite_hdr(self, fname_out_envi):
#        '''
#        Replaces .hdr file with self.metadata items
#        '''
#        _, ext = os.path.splitext(fname_out_envi)
#        if ext != '.hdr':
#            fname_out_envi += '.hdr'
#
#        with open(fname_out_envi, 'w') as f:
#            f.write('ENVI\n')
#            for key, val in sorted(self.metadata.items()):
#                f.write('{0} = {1}\n'.format(key, val))
#
#    def _save_file_setup(self, base_dir_out=None, folder_name='band_math'):
#        '''
#        Basic setup items when saving manipulated image files to disk
#
#        Parameters:
#            base_dir_out (`str`): The base
#            folder_name (`str`):
#        '''
#        if base_dir_out is None:
#            base_dir_out = os.path.join(self.base_dir, folder_name)
#        if not os.path.isdir(base_dir_out):
#            os.mkdir(base_dir_out)
#
#        if self.name_plot is not None:
#            name_print = self.name_plot
#        else:
#            name_print = self.name_short
#        return base_dir_out, name_print

#    def _write_envi(self, array, fname_out, geotransform_out, name=None,
#                    interleave='bip', rewrite_hdr=True):
#        '''
#        Writes datacube to ENVI file
#
#        Parameters:
#            array (numpy array): input image cube; must be in band sequential
#                (x, y, z) format to properly save base on interleave indicated
#            fname_out (`str`):
#            geotransform_out (`GDAL geotransform`):
#            name (`str`):
#            interleave (`str`):
#            rewrite_hdr (`bool`): indicates if header file should be replaced
#                by self.metadata items (default=True)
#        '''
#        try:
#            ysize, xsize, bands = array.shape
#        except ValueError as e:
#            ysize, xsize = array.shape
#            bands = 1
#
#        base_name, ext = os.path.splitext(fname_out)
#        if ext != '.' + interleave:
#            fname_out = base_name + '.' + interleave
#
#        if interleave.lower == 'bip':
#            interleave_str = 'INTERLEAVE=BIP'
#        elif interleave.lower == 'bil':
#            interleave_str = 'INTERLEAVE=BIL'
#        elif interleave.lower == 'bsq':
#            interleave_str = 'INTERLEAVE=BSQ'
#
#        drv = gdal.GetDriverByName('ENVI')
#        drv.Register()
#        ds_out = drv.Create(fname_out, xsize, ysize, bands, gdal.GDT_Float32,
#                            ['SUFFIX=ADD', interleave_str])
#        ds_out.SetGeoTransform(geotransform_out)
#        ds_out.SetProjection(self.projection)
#
#        for band in range(bands):
#            if bands == 1:
#                array_band = array[:, :]
#            else:
#                array_band = array[:, :, band]
#            band_out = ds_out.GetRasterBand(band + 1)
#            if name is None:
#                band_out.SetDescription(str(band + 1))
#            else:
#                band_out.SetDescription(name)
#            band_out.WriteArray(array_band)
#            band_out = None
#        ds_out.FlushCache()
#        drv = None
#        ds_out = None
#        if rewrite_hdr is True:
#            self._rewrite_hdr(fname_out)
#
#    def _write_envi_spy(self, fname_out, df_mean, df_std, interleave='bip',
#                        dtype=np.float32, byteorder=0, force=True):
#        '''
#        Writes spectra to ENVI file
#        '''
#        if os.path.splitext(fname_out)[1] != '.hdr':
#            fname_out = fname_out + '.hdr'
#
#        metadata = self.metadata
#        metadata = self._del_meta_item(metadata, 'map info')
#        metadata = self._del_meta_item(metadata, 'history')
#
#        metadata = self._del_meta_item(metadata, 'original cube file')
#        metadata = self._del_meta_item(metadata, 'pointlist')
#        metadata = self._del_meta_item(metadata, 'boundary')
#        metadata = self._del_meta_item(metadata, 'label')
#
#        band_names = ', '.join(str(e) for e in list(self.meta_bands.keys()))
#        metadata['band names'] = '{' + band_names + '}'
#
#        std = df_std.to_dict()
#        stdev = ', '.join(str(e) for e in list(std.values()))
#        metadata['stdev'] = '{' + stdev + '}'
#
#        array_mean = df_mean.to_numpy()
#        array = array_mean.reshape(1, 1, len(df_mean))
#        envi.save_image(fname_out, array, interleave=interleave, dtype=dtype,
#                        byteorder=byteorder, metadata=metadata, force=force,
#                        ext=None)
#
#    def _write_tif(self, array_img_crop, fname_out_tif, projection_out,
#                   geotransform_out):
#        '''
#        Writes RGB geotif to file
#        '''
#        drv = gdal.GetDriverByName('GTiff')
#        drv.Register()
#        ysize, xsize, bands = array_img_crop.shape
#        tif_out = drv.Create(fname_out_tif, xsize, ysize, 3, gdal.GDT_Float32)
#        tif_out.SetProjection(projection_out)
#        tif_out.SetGeoTransform(geotransform_out)
#
#        band_b = self._get_band(460)[0]
#        band_g = self._get_band(550)[0]
#        band_r = self._get_band(640)[0]
#        band_list = [band_b, band_g, band_r]
#
#        array_img = None
#        for idx, band in enumerate(band_list):
#            # for whatever reason, GDAL needs N/S pixels (rows) to be flipped
#            array_band = np.flip(array_img_crop[:, :, band-1], axis=0)
#            band_out = tif_out.GetRasterBand(idx + 1)
#            band_out.WriteArray(array_band)
#            if array_img is None:
#                array_img = array_band
#            else:
#                array_img = np.dstack((array_img, array_band))  # stacks bands
#            band_out = None
#        tif_out.FlushCache()
#        drv = None
#        tif_out = None
#        self.show_img(array_img_crop, band_r=band_r, band_g=band_g,
#                      band_b=band_b)
#
#    def read_cube(self, fname_in, name_long='-Unit Conversion Utility',
#                  plot_name=False):
#        '''
#        Reads in a hyperspectral datacube
#
#        fname_in (str): filename of datacube to be read
#        name_long (str): Spectronon processing appends processing names to
#            the filenames; this indicates those processing names that are
#            repetitive and can be deleted from the filename following
#            processing.
#        plot_name (bool): Indicates whether image (and its filename) is for an
#            individual plot (True), or for many plots (False) (default: False).
#        '''
#        self.fname_in = fname_in
#        self.base_dir = os.path.split(fname_in)[0]
#        self.name_long = name_long
#        base_name = os.path.basename(self.fname_in)
#        name_short = base_name[:base_name.find(self.name_long)]
#        self.name_short = name_short
#        self.plot_name = plot_name
#
#        if name_short[-1] == '_' or name_short[-1] == '-':
#            name_short = name_short[:-1]
#        if plot_name is True:
##            self.name_plot = 'plot' + name_short[name_short.rfind('_'):]
#            self.name_plot = 'plot' + name_short[name_short.find('_'):]
#            plot = name_short[name_short.rfind('_')+1:]
#            self.plot = plot
#        else:
#            self.name_plot = None
#            self.plot = None
#
##        self._read_envi_gdal()
#        self._read_envi_hdr()
#        self._read_envi_spy()
#
#    def read_spec(self, fname_in=None):
#        if fname_in is None:
#            fname_in = self.fname_in
#        else:
#            self.fname_in = fname_in
#        self.base_dir = os.path.split(fname_in)[0]
#
##        self._read_envi_gdal()
#        self._read_envi_hdr(fname_in)
#        self._read_envi_spy()
#        self.array_smooth = self._smooth_image()
#
#    def write_cube(self, hdr_file, spyfile, dtype=np.float32,
#                   force=False, ext=None, interleave='bip', byteorder=None,
#                   metadata=None):
#        '''
#        Wrapper function that accesses the Spectral Python package to save a
#        datacube to file.
#
#        Parameters:
#            hdr_file (`str`): Header file path (with the '.hdr' extension).
#            spyfile (`SpyFile` object or `numpy.ndarray`): The hyperspectral
#                data cube to save. If `numpy.ndarray`, then metadata (`dict`)
#                should also be passed.
#            dtype (`numpy.dtype` or `str`): The data type with which to store
#                the image. For example, to store the image in 16-bit unsigned
#                integer format, the argument could be any of numpy.uint16,
#                'u2', 'uint16', or 'H' (default=np.float32).
#            force (`bool`): If `hdr_file` or its associated image file exist,
#                `force=True` will overwrite the files; otherwise, an exception
#                will be raised if either file exists (default=False).
#            ext (`str`): The extension to use for saving the image file; if not
#                specified, a default extension is determined based on the
#                `interleave`. For example, if `interleave`='bip', then `ext` is
#                set to 'bip' as well. If `ext` is an empty string, the image
#                file will have the same name as the .hdr, but without the
#                '.hdr' extension.
#            interleave (`str`): The band interleave format to use for writing
#                the file; `interleave` should be one of 'bil', 'bip', or 'bsq'
#                (default='bip').
#            byteorder (`int` or `str`): Specifies the byte order (endian-ness)
#                of the data as written to disk. For little endian, this value
#                should be either 0 or 'little'. For big endian, it should be
#                either 1 or 'big'. If not specified, native byte order will be
#                used (default=None).
#            metadata (`dict`): Metadata to write to the ENVI .hdr file
#                describing the hyperspectral data cube being saved. If
#                `SpyFile` object is passed to `cube_spy`, `metadata` will
#                overwrite any existing metadata stored by the `SpyFile` object
#                (default=None).
#        '''
#        if ext is None:
#            ext = '.' + interleave
#        if metadata is None and isinstance(spyfile, SpyFile.SpyFile):
#            metadata = spyfile.metadata
#        envi.save_image(hdr_file, spyfile, dtype=dtype, force=force, ext=ext,
#                        interleave=interleave, byteorder=byteorder,
#                        metadata=metadata)
#
#    def write_spec_spy(self, fname_out, df_mean, df_std, interleave='bip',
#                       dtype=np.float32, byteorder=0, force=True):
#        '''
#        Writes spectra to ENVI file
#        '''
#        if os.path.splitext(fname_out)[1] != '.hdr':
#            fname_out = fname_out + '.hdr'
#
#        metadata = self.metadata
#        metadata = self._del_meta_item(metadata, 'map info')
#        metadata = self._del_meta_item(metadata, 'history')
#
#        metadata = self._del_meta_item(metadata, 'original cube file')
#        metadata = self._del_meta_item(metadata, 'pointlist')
#        metadata = self._del_meta_item(metadata, 'boundary')
#        metadata = self._del_meta_item(metadata, 'label')
#
#        band_names = ', '.join(str(e) for e in list(self.meta_bands.keys()))
#        metadata['band names'] = '{' + band_names + '}'
#
#        std = df_std.to_dict()
#        stdev = ', '.join(str(e) for e in list(std.values()))
#        metadata['stdev'] = '{' + stdev + '}'
#
#        array_mean = df_mean.to_numpy()
#        array = array_mean.reshape(1, 1, len(df_mean))
#        envi.save_image(fname_out, array, interleave=interleave, dtype=dtype,
#                        byteorder=byteorder, metadata=metadata, force=force,
#                        ext=None)
#
#
#class HS_tools(object):
#    '''
#    Some basic tools for retrieving particular bands, the wavelengths they
#    represent, and their order in the data array.
#    '''
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
#
#    def _get_band(self, target):
#        '''
#        Returns band number of closest target wavelength
#        band = self._get_band(703) returns 151 (i.e., band 151)
#
#        Parameters:
#            target (`int` or `float`): the target wavelength to retrive band
#                number for (required).
#        '''
#        val_target = min(list(self.meta_bands.values()),
#                         key=lambda x: abs(x-target))
#        key_band = list(self.meta_bands.keys())[sorted(list(
#                self.meta_bands.values())).index(val_target)]
#        key_wavelength = sorted(list(self.meta_bands.values()))[key_band-1]
#        return key_band, key_wavelength
#
#    def _get_band_index(self, band_num):
#        '''
#        Subtracts 1 from each number in band list and returns list of band
#        indexes
#        '''
#        if isinstance(band_num, list):
#            band_num = np.array(band_num)
#            band_idx = list(band_num - 1)
#        else:
#            band_idx = band_num - 1
#        return band_idx
#
#    def _get_band_info_consolidate(self, b1, b2):
#        '''
#        Gets band number and wavelength information
#        '''
#        if isinstance(b1, list):
#            band1 = []
#            wl1 = []
#            for band in b1:
#                band_i, wl_i = self._get_band(band)
#                band1.append(band_i)
#                wl1.append(wl_i)
#            wl1 = np.mean(wl1)
#        else:
#            band1, wl1 = self._get_band(b1)
#        if isinstance(b2, list):
#            band2 = []
#            wl2 = []
#            for band in b2:
#                band_i, wl_i = self._get_band(band)
#                band2.append(band_i)
#                wl2.append(wl_i)
#            wl2 = np.mean(wl2)
#        else:
#            band2, wl2 = self._get_band(b2)
#        return band1, band2, wl1, wl2
#
#    def _get_band_mean(self, img_array, band_num):
#        '''
#        Gets the mean value from a list of bands
#
#        Parameters:
#            img_array (numpy array): image to evaluate
#            band_num (int, list): band number(s) to determine mean value for
#        '''
#        band_idx = self._get_band_index(band_num)
#        if isinstance(band_idx, list):
#            array_band = np.mean(img_array[:, :, band_idx], axis=2)
#        else:
#            array_band = img_array[:, :, band_idx]
#        return array_band
#
#    def _get_band_num(self, band_idx):
#        '''
#        Adds 1 to each number in band list and returns list of band
#        numbers
#        '''
#        if isinstance(band_idx, list):
#            band_idx = np.array(band_idx)
#            band_num = list(band_idx + 1)
#        else:
#            band_num = band_idx + 1
#        return band_num
#
#    def _get_band_range(self, range_wl, index=True):
#        '''
#        Gets all band indexes with the given minimum and maximum wavelengths
#
#        Parameters:
#            range_wl (list): the minimum and maximum wavelength to consider;
#                values should be `int` or `float`.
#            index (bool): Indicates whether to return the band number (min=1)
#                or to return index number (min=0) (default: True)
#        '''
#        band_min, wl_min = self._get_band(range_wl[0])
#        band_max, wl_max = self._get_band(range_wl[1])
#        if wl_min < range_wl[0]:
#            band_min += 1
#        if wl_max > range_wl[1]:
#            band_max -= 1
#        if index is True:
#            band_min = self._get_band_index(band_min)
#            band_max = self._get_band_index(band_max)
#        band_list = [x for x in range(band_min, band_max+1)]
#        return band_list

#    def _get_meta_set(self, meta_set, idx=None):
#        '''
#        Reads a value from metadata "set" (dict-like) based on the index
#
#        Parameters:
#            meta_set (`str`): the string representation of the metadata set
#            idx (`int`): index to be read
#        '''
#        meta_set_list = meta_set[1:-1].split(",")
#        meta_set_str = []
#        for item in meta_set_list:
#            if str(item)[::-1].find('.') == -1:
#                try:
#                    meta_set_str.append(int(item))
#                except ValueError as e:
#                    if item[0] == ' ':
#                        meta_set_str.append(item[1:])
#                    else:
#                        meta_set_str.append(item)
#            else:
#                try:
#                    meta_set_str.append(float(item))
#                except ValueError as e:
#                    if item[0] == ' ':
#                        meta_set_str.append(item[1:])
#                    else:
#                        meta_set_str.append(item)
#        if idx is None:
#            return meta_set_str  # return the whole thing
#        else:
#            return meta_set_str[idx]
#
#    def show_img(self, array_img, band_r=120, band_g=76, band_b=32,
#                 inline=True):
#        '''
#        Displays the RGB bands
#        '''
#        if inline is True:
#            get_ipython().run_line_magic('matplotlib', 'inline')
#        else:
#            get_ipython().run_line_magic('matplotlib', 'auto')
#        if len(array_img.shape) == 2:
#            n_bands = 1
#            ysize, xsize = array_img.shape
#        elif len(array_img.shape) == 3:
#            ysize, xsize, n_bands = array_img.shape
#        else:
#            raise NotImplementedError('Only 2-D and 3-D arrays can be '
#                                      'displayed.')
#        if n_bands >= 3:
#            try:
#                plt.imshow(array_img, (band_r, band_g, band_b))
#            except ValueError as err:
#                plt.imshow(array_img[:, :, [band_r, band_g, band_b]]*3.5)
##            array_img_out = array_img[:, :, [band_r, band_g, band_b]]
##            array_img_out *= 3.5  # Images are very dark without this
#
#        else:
#            plt.imshow(array_img)
##            array_img_out = array_img
##            plt.imshow(array_img_out)
#        plt.show()
#        print('\n')
