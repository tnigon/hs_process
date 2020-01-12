# -*- coding: utf-8 -*-
import ast
from matplotlib import pyplot as plt
import numpy as np
import os
from osgeo import gdal
from osgeo import gdalconst
#from osgeo import ogr
import pandas as pd
import re
import seaborn as sns
import spectral.io.envi as envi
import spectral.io.spyfile as SpyFile
import sys


class defaults:
    '''
    Class containing all defaults for writing an ENVI datacube to file.

    Parameters:
        dtype (``numpy.dtype`` or ``str``): The data type with which to store
            the image. For example, to store the image in 16-bit unsigned
            integer format, the argument could be any of numpy.uint16,
            'u2', 'uint16', or 'H' (default=np.float32).
        force (``bool``): If ``hdr_file`` or its associated image file exist,
            ``force=True`` will overwrite the files; otherwise, an exception
            will be raised if either file exists (default=False).
        ext (``str``): The extension to use for saving the image file; if not
            specified, a default extension is determined based on the
            ``interleave``. For example, if ``interleave``='bip', then ``ext`` is
            set to 'bip' as well. If ``ext`` is an empty string, the image
            file will have the same name as the .hdr, but without the
            '.hdr' extension (default: None).
        interleave (``str``): The band interleave format to use for writing
            the file; ``interleave`` should be one of 'bil', 'bip', or 'bsq'
            (default='bip').
        byteorder (``int`` or ``str``): Specifies the byte order (endian-ness)
            of the data as written to disk. For little endian, this value
            should be either 0 or 'little'. For big endian, it should be
            either 1 or 'big'. If not specified, native byte order will be
            used (default=None).
    '''
    dtype = np.float32
    force = False
    ext = ''
    interleave = 'bip'
    byteorder = 0

    envi_write = {
            'dtype': np.float32,
            'force': False,
            'ext': '',
            'interleave': 'bip',
            'byteorder': 0}

    spat_crop_cols = {
            'directory': 'directory',
            'fname': 'fname',
            'name_short': 'name_short',
            'name_long': 'name_long',
            'ext': 'ext',
            'pix_e_ul': 'pix_e_ul',
            'pix_n_ul': 'pix_n_ul',
            'plot_id': 'plot_id',
            'alley_size_e_m': 'alley_size_e_m',
            'alley_size_n_m': 'alley_size_n_m',
            'alley_size_e_pix': 'alley_size_e_pix',
            'alley_size_n_pix': 'alley_size_n_pix',
            'buf_e_m': 'buf_e_m',
            'buf_n_m': 'buf_n_m',
            'buf_e_pix': 'buf_e_pix',
            'buf_n_pix': 'buf_n_pix',
            'crop_e_m': 'crop_e_m',
            'crop_n_m': 'crop_n_m',
            'crop_e_pix': 'crop_e_pix',
            'crop_n_pix': 'crop_n_pix',
            'n_plots_x': 'n_plots_x',
            'n_plots_y': 'n_plots_y',
            'n_plots': 'n_plots'}

    crop_defaults = {
            'directory': None,
            'name_short': None,
            'name_long': None,
            'ext': 'bip',
            'pix_e_ul': 0,
            'pix_n_ul': 0,
            'alley_size_e_pix': None,  # set to `None` because should be set
            'alley_size_n_pix': None,  # intentionally
            'alley_size_e_m': None,
            'alley_size_n_m': None,
            'crop_e_pix': 90,
            'crop_n_pix': 120,
            'crop_e_m': None,
            'crop_n_m': None,
            'buf_e_pix': 0,
            'buf_n_pix': 0,
            'buf_e_m': None,
            'buf_n_m': None,
            'plot_id': None}


class hsio(object):
    '''
    Class for reading and writing hyperspectral data files and accessing,
    interpreting, and modifying its associated metadata.

    TODO: Create a temporary Spyfile using envi.create_imamge() and saving to a
        temporary location. This can be used to hold intermediate SpyFiles
        without actually saving them to disk.. (good idea?)
    '''
    def __init__(self, fname_in=None, name_long=None, name_plot=None,
                 name_short=None, str_plot='plot_', individual_plot=False,
                 fname_hdr_spec=None):
        '''
        Parameters:
            fname_in (``str``, optional): The filename of the image datacube to
                be read in initially.
        '''
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
        self.spyfile = None
        self.spyfile_spec = None
        self.tools = None

        if fname_in is not None:
            if os.path.splitext(fname_in)[1] != '.hdr':
                self.fname_hdr = fname_in + '.hdr'
            else:
                self.fname_hdr = fname_in
                self.fname_in = os.path.splitext(fname_in)[0]
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

    def _del_meta_item(self, metadata, key):
        '''
        Deletes metadata item from SpyFile object.

        Parameters:
            metadata (``dict``): dictionary of the metadata
            key (``str``): dictionary key to delete

        Returns:
            metadata (``dict``): Dictionary containing the modified metadata.
        '''
        msg = ('Please be sure to base a metadata dictionary.')
        assert isinstance(metadata, dict), msg
        try:
#            del metadata[key]
            _ = metadata.pop(key, None)
#            val = None
        except KeyError:
            print('{0} not a valid key in input dictionary.'.format(key))
        return metadata

    def _get_meta_set(self, meta_set, idx=None):
        '''
        Reads metadata "set" (i.e., string representation of a Python set;
        common in .hdr files), taking care to remove leading and trailing
        spaces.

        Parameters:
            meta_set (``str``): the string representation of the metadata set
            idx (``int``): index to be read; if ``None``, the whole list is
                returned (default: ``None``).

        Returns:
            metadata_list (``list`` or ``str``): List of metadata set items (as
                ``str``), or if idx is not ``None``, the item in the position
                described by ``idx``.
        '''
        if isinstance(meta_set, str):
            meta_set_list = meta_set[1:-1].split(",")
        else:
            meta_set_list = meta_set.copy()
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
            meta_set (``str``): the string representation of the metadata set
            idx (``int``): index to be modified; if ``None``, the whole meta_set is
                returned (default: ``None``).
            value (``float``, ``int``, or ``str``): value to replace at idx

        Returns:
            set_str (``str``):
        '''
        metadata_list = self._get_meta_set(meta_set, idx=None)
        metadata_list[idx] = str(value)
        set_str = '{' + ', '.join(str(x) for x in metadata_list) + '}'
        return set_str

    def _parse_fname(self, fname_hdr=None, str_plot='plot_', overwrite=True):
        '''
        Parses the filename for ``name_long`` (text after the first dash,
        inclusive), ``name_short`` (text before the first dash), and ``name_plot``
        (numeric text following ``str_plot``).

        Parameters:
            fname_hdr (``str``): input filename.
            str_plot (``str``): text to search for that precedes the numeric text
                that describes the plot number.
            overwrite (``bool``): whether the class instances of ``name_long``,
                ``name_short``, and ``name_plot`` should be overwritten based on
                ``fname_in`` (default: ``True``).
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
            spec (``bool``): Whether the file to be read is an image (``False``) or
                a spectrum (``True``; default: ``False``).
        '''
        if spec is False:
#            fname_hdr = self.fname_in + '.hdr'
            fname_hdr = self.fname_hdr
            try:
                self.spyfile = envi.open(fname_hdr)
            except envi.MissingEnviHeaderParameter as e:  # Resonon excludes
                err = str(e)
                key = err[err.find('"') + 1:err.rfind('"')]
                if key == 'byte order':
                    self._append_hdr_fname(fname_hdr, key, 0)
                else:
                    print(err)
                self.spyfile = envi.open(fname_hdr)
            self.tools = hstools(self.spyfile)
        else:
            fname_hdr_spec = self.fname_hdr_spec
            try:
                self.spyfile_spec = envi.open(fname_hdr_spec)
            except envi.MissingEnviHeaderParameter as e:  # Resonon excludes
                err = str(e)
                key = err[err.find('"') + 1:err.rfind('"')]
                if key == 'byte order':
                    self._append_hdr_fname(fname_hdr_spec, key, 0)
                else:
                    print(err)
                self.spyfile_spec = envi.open(fname_hdr_spec)
            self.tools = hstools(self.spyfile_spec)

    def _read_envi_gdal(self, fname_in=None):
        '''
        Reads and ENVI file via GDAL

        Parameters:
            fname_in (``str``): filename of the ENVI file to read (not the .hdr;
                default: ``None``).

        Returns:
            img_ds (``GDAL object``): GDAL dataset containing the image
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
            fname_hdr (``str``): filename of .hdr file

        Returns:
            metadata (``dict``): dictionary of the metadata
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
            fname_hdr (``str``): filename of .hdr file to write (default:
                ``None``).
            metadata (``dict``): dictionary of the metadata (default: ``None``).
        '''
        if fname_hdr is None:
            fname_hdr = self.fname_in + '.hdr'
        if metadata is None:
            metadata = self.spyfile.metadata
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
        Function for safely returning an empty string (e.g., ``None``).

        Parameters:
            s (``str`` or ``None``): the variable that may contain a string.
        '''
        if s is None:
            return ''
        return str('-' + s)

    def read_cube(self, fname_hdr=None, name_long=None, name_plot=None,
                  name_short=None, individual_plot=False, overwrite=True):
        '''
        Reads in a hyperspectral datacube

        Parameters:
            fname_hdr (str): filename of datacube to be read (default: ``None``).
            name_long (str): Spectronon processing appends processing names to
                the filenames; this indicates those processing names that are
                repetitive and can be deleted from the filename following
                processing (default: ``None``).
            name_plot (``str``): numeric text that describes the plot number
                (default: ``None``).
            name_short (``str``): The base name of the image file (see note above
                about ``name_long``; default: ``None``).
            individual_plot (``bool``): Indicates whether image (and its
                filename) is for an individual plot (``True``), or for many plots
                (``False``; default: ``False``).
            overwrite (``bool``): Whether to overwrite any of the previous
                user-passed variables, including ``name_long``, ``name_plot``, and
                ``name_short``; any of the current user-passed variables will
                overwrite previous ones whether ``overwrite`` is ``True`` or
                ``False`` (default: ``False``).
        '''
        # Basically resets static __init__ variables for the new filename
        # If variables are already set and overwrite is False, they will remain
        # the same; if variables are set and overwrite is True, they will be
        # overwritten
#        self._parse_fname(fname_hdr, self.str_plot, overwrite=overwrite)

#        if os.path.splitext(fname_hdr)[1] != '.hdr':
#            fname_hdr = fname_hdr + '.hdr'
#            self.fname_hdr = fname_hdr
#
#        # The following ensures that user-passed variables have priority
#        if not os.path.isfile(fname_hdr):
#            fname_hdr = self.fname_in
#        if name_long is not None:
#            self.name_long = name_long
#        if name_plot is not None:
#            self.name_plot = name_plot
#        if name_short is not None:
#            self.name_short = name_short

        if os.path.splitext(fname_hdr)[1] != '.hdr':
            fname_hdr = fname_hdr + '.hdr'
        assert os.path.isfile(fname_hdr), 'Could not find .hdr file.'
        self.fname_hdr = fname_hdr
        self.base_dir = os.path.dirname(fname_hdr)

        self.individual_plot = individual_plot
        self._read_envi()

        basename = os.path.basename(fname_hdr)
        self.name_short = basename[:basename.find('-', basename.rfind('_'))]
        self.name_long = basename[basename.find('-', basename.rfind('_')):]
        self.name_plot = self.name_short.rsplit('_', 1)[1]

    def read_spec(self, fname_hdr_spec):
        '''
        Reads in a hyperspectral spectrum file

        Parameters:
            fname_hdr_spec (``str``): filename of spectra to be read.
        '''
        if os.path.splitext(fname_hdr_spec)[1] != '.hdr':
            fname_hdr_spec = fname_hdr_spec + '.hdr'
        assert os.path.isfile(fname_hdr_spec), 'Could not find .hdr file.'
        self.fname_hdr_spec = fname_hdr_spec
        self.base_dir_spec = os.path.dirname(fname_hdr_spec)
        self._read_envi(spec=True)
        basename = os.path.basename(fname_hdr_spec)
        self.name_short = basename[:basename.find('-', basename.rfind('_'))]
        self.name_long = basename[basename.find('-', basename.rfind('_')):]
        self.name_plot = self.name_short.rsplit('_', 1)[1]

    def set_io_defaults(self, dtype=False, force=None, ext=False,
                        interleave=False, byteorder=False):
        '''
        Sets any of the ENVI file writing parameters to ``hsio``; if any
        parameter is left unchanged from its default, it will remain as-is
        (it will not be set).

        Parameters:
            dtype (``numpy.dtype`` or ``str``): The data type with which to store
                the image. For example, to store the image in 16-bit unsigned
                integer format, the argument could be any of numpy.uint16,
                'u2', 'uint16', or 'H' (default=``False``).
            force (``bool``): If ``hdr_file`` or its associated image file exist,
                ``force=True`` will overwrite the files; otherwise, an exception
                will be raised if either file exists (default=``None``).
            ext (``str``): The extension to use for saving the image file; if not
                specified, a default extension is determined based on the
                ``interleave``. For example, if ``interleave``='bip', then ``ext`` is
                set to 'bip' as well. If ``ext`` is an empty string, the image
                file will have the same name as the .hdr, but without the
                '.hdr' extension (default: ``False``).
            interleave (``str``): The band interleave format to use for writing
                the file; ``interleave`` should be one of 'bil', 'bip', or 'bsq'
                (default=``False``).
            byteorder (``int`` or ``str``): Specifies the byte order (endian-ness)
                of the data as written to disk. For little endian, this value
                should be either 0 or 'little'. For big endian, it should be
                either 1 or 'big'. If not specified, native byte order will be
                used (default=``False``).
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
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The data cube to
                display; if ``None``, loads from ``self.spyfile`` (default:
                ``None``).
            band_r (``int``): Band to display on the red channel (default: 120)
            band_g (``int``): Band to display on the green channel (default: 76)
            band_b (``int``): Band to display on the blue channel (default: 32)
            inline (``bool``): If ``True``, displays in the IPython console; else
                displays in a pop-out window (default: ``True``).
        '''
        if inline is True:
            get_ipython().run_line_magic('matplotlib', 'inline')
        else:
            get_ipython().run_line_magic('matplotlib', 'auto')

        if spyfile is None:
            spyfile = self.spyfile
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

    def write_cube(self, fname_hdr, spyfile, metadata=None, dtype=None,
                   force=None, ext=None, interleave=None, byteorder=None):
        '''
        Wrapper function that accesses the Spectral Python package to save a
        datacube to file.

        Parameters:
            fname_hdr (``str``): Output header file path (with the '.hdr'
                extension).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The hyperspectral
                datacube to save. If ``numpy.ndarray``, then metadata (``dict``)
                should also be passed.
            metadata (``dict``): Metadata to write to the ENVI .hdr file
                describing the hyperspectral data cube being saved. If
                ``SpyFile`` object is passed to ``spyfile``, ``metadata`` will
                overwrite any existing metadata stored by the ``SpyFile`` object
                (default=None).
            dtype (``numpy.dtype`` or ``str``): The data type with which to store
                the image. For example, to store the image in 16-bit unsigned
                integer format, the argument could be any of numpy.uint16,
                'u2', 'uint16', or 'H' (default=None).
            force (``bool``): If ``hdr_file`` or its associated image file exist,
                ``force=True`` will overwrite the files; otherwise, an exception
                will be raised if either file exists (default=None).
            ext (``str``): The extension to use for saving the image file; if not
                specified, a default extension is determined based on the
                ``interleave``. For example, if ``interleave``='bip', then ``ext`` is
                set to 'bip' as well. If ``ext`` is an empty string, the image
                file will have the same name as the .hdr, but without the
                '.hdr' extension (default: None).
            interleave (``str``): The band interleave format to use for writing
                the file; ``interleave`` should be one of 'bil', 'bip', or 'bsq'
                (default=None).
            byteorder (``int`` or ``str``): Specifies the byte order (endian-ness)
                of the data as written to disk. For little endian, this value
                should be either 0 or 'little'. For big endian, it should be
                either 1 or 'big'. If not specified, native byte order will be
                used (default=None).

        Note:
            If ``dtype``, ``force``, ``ext``, ``interleave``, and ``byteorder`` are not
            passed, default values will be pulled from ``hsio.defaults``. Thus,
            ``hsio.defaults`` can be modified prior to calling
            ``hsio.write_cube()`` to avoid having to pass each of thes parameters
            in the ``hsio.write_cube()`` function (see the
            ``hsio.set_io_defaults()`` function for support on setting these
            defaults and for more information on the parameters). Each of these
            parameters are passed directly to the Spectral Python
            ``envi.save_image()`` function. For more information, please refer to
            the Spectral Python documentation.
        '''
        if dtype is None:
            dtype = self.defaults.dtype
        if force is None:
            force = self.defaults.force
        if ext is None:
            ext = self.defaults.ext
        if interleave is None:
            interleave = self.defaults.interleave
        if byteorder is None:
            byteorder = self.defaults.byteorder

        if metadata is None and isinstance(spyfile, SpyFile.SpyFile):
            metadata = spyfile.metadata
        elif metadata is None and isinstance(spyfile, np.ndarray):
            raise TypeError('`spyfile` of type `numpy.ndarray` was passed, so '
                            '`metadata` must not be `None`.')

        metadata['interleave'] = interleave
        if os.path.splitext(fname_hdr)[1] != '.hdr':
            fname_hdr = fname_hdr + '.hdr'
        metadata = self.tools.clean_md_sets(metadata=metadata)
        envi.save_image(fname_hdr, spyfile, dtype=dtype, force=force, ext=ext,
                        interleave=interleave, byteorder=byteorder,
                        metadata=metadata)

    def write_spec(self, hdr_file, df_mean, df_std, dtype=np.float32,
                   force=True, ext='.spec', interleave='bip', byteorder=0,
                   metadata=None):
        '''
        Wrapper function that accesses the Spectral Python package to save a
        single spectra to file.

        Parameters:
            hdr_file (``str``): Output header file path (with the '.hdr'
                extension).
            df_mean (``pandas.Series`` or ``numpy.ndarray``): Mean spectra, stored as a df row,
                where columns are the bands.
            df_std (``pandas.Series`` or ``numpy.ndarray``): Standard deviation of each spectra,
                stored as a df row, where columns are the bands. This will be
                saved to the .hdr file.
            dtype (``numpy.dtype`` or ``str``): The data type with which to store
                the image. For example, to store the image in 16-bit unsigned
                integer format, the argument could be any of numpy.uint16,
                'u2', 'uint16', or 'H' (default=np.float32).
            force (``bool``): If ``hdr_file`` or its associated image file exist,
                ``force=True`` will overwrite the files; otherwise, an exception
                will be raised if either file exists (default=False).
            ext (``str``): The extension to use for saving the image file; if not
                specified, a default extension is determined based on the
                ``interleave``. For example, if ``interleave``='bip', then ``ext`` is
                set to 'bip' as well. If ``ext`` is an empty string, the image
                file will have the same name as the .hdr, but without the
                '.hdr' extension (default: '.spec').
            interleave (``str``): The band interleave format to use for writing
                the file; ``interleave`` should be one of 'bil', 'bip', or 'bsq'
                (default='bip').
            byteorder (``int`` or ``str``): Specifies the byte order (endian-ness)
                of the data as written to disk. For little endian, this value
                should be either 0 or 'little'. For big endian, it should be
                either 1 or 'big'. If not specified, native byte order will be
                used (default=None).
            metadata (``dict``): Metadata to write to the ENVI .hdr file
                describing the spectra being saved; if ``None``, will try to pull
                metadata template from self.spyfile.metadata (default=None).
        '''
        if ext is None:
            ext = '.' + interleave
        if metadata is None:
            metadata = self.spyfile_spec.metadata
        if os.path.splitext(hdr_file)[1] != '.hdr':
            hdr_file = hdr_file + '.hdr'
        metadata = self._del_meta_item(metadata, 'map info')
        metadata = self._del_meta_item(metadata, 'coordinate system string')
#        metadata = self._del_meta_item(metadata, 'history')
#        metadata = self._del_meta_item(metadata, 'original cube file')
#        metadata = self._del_meta_item(metadata, 'pointlist')
        metadata = self._del_meta_item(metadata, 'boundary')
        metadata = self._del_meta_item(metadata, 'label')

        if 'band names' not in metadata.keys():
            metadata['band names'] = '{' + ', '.join(str(e) for e in list(
                    self.tools.meta_bands.keys())) + '}'
        if 'wavelength' not in metadata.keys():
            metadata['wavelength'] = '{' + ', '.join(str(e) for e in list(
                    self.tools.meta_bands.values())) + '}'
        if isinstance(df_std, np.ndarray):
            df_std = pd.Series(df_std)
        std = df_std.to_dict()
        metadata['stdev'] = '{' + ', '.join(str(e) for e in list(
                std.values())) + '}'
        metadata['label'] = os.path.basename(os.path.splitext(hdr_file)[0])
        metadata = self.tools.clean_md_sets(metadata=metadata)
        try:
            self.spyfile_spec.metadata = metadata
        except AttributeError as err:
            pass
        if isinstance(df_mean, np.ndarray):
            array_mean = df_mean.copy()
        else:
            array_mean = df_mean.to_numpy()
        array = array_mean.reshape(1, 1, len(df_mean))
        envi.save_image(hdr_file, array, interleave=interleave, dtype=dtype,
                        byteorder=byteorder, metadata=metadata, force=force,
                        ext=ext)

    def write_tif(self, fname_tif, spyfile=None, fname_in=None,
                  projection_out=None, geotransform_out=None, metadata=None,
                  inline=True):
        '''
        Wrapper function that accesses the GDAL Python package to save a
        small datacube subset (i.e., three bands or less) to file.

        Parameters:
            fname_tif (``str``): Output image file path (with the '.tif'
                extension).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The data cube to
                save. If ``numpy.ndarray``, then metadata (``dict``) should also be
                passed.
            fname_in (``str``, optional): The filename of the image datacube to
                be read in initially.
            projection_out (``str``): (default: ``self.projection_out``)
            geotransform_out (``str``): (default: ``self.geotransform_out``)
            metadata (``dict``): Metadata information; if geotransform_out is not
                passed, "map info" is accessed from ``metadata`` and
                geotransform_out is created from that "map info".

        TOOD:
            Use rasterio package instead of GDAL
        '''
        msg = ('The directory passed in `fname_tif` does not exist. Please be '
               'sure to create the directory prior to writing the geotif.\n'
               'Try:\n'
               'os.mkdir(os.path.dirname(fname_tif))'
               ''.format(os.path.dirname(fname_tif)))
        assert os.path.isdir(os.path.dirname(fname_tif)) is True, msg
        if spyfile is None:
            spyfile = self.spyfile
        if isinstance(spyfile, SpyFile.SpyFile):
            array = spyfile.load()
        else:
            assert isinstance(spyfile, np.ndarray)
            array = spyfile
        if fname_in is not None:
            self.fname_in = fname_in
            if os.path.splitext(fname_in)[1] != '.hdr':
                self.fname_hdr = fname_in + '.hdr'
            else:
                self.fname_hdr = fname_in
                self.fname_in = os.path.splitext(fname_in)[0]
            self.read_cube(fname_hdr=self.fname_hdr, name_long=self.name_long,
                           name_plot=self.name_plot,
                           name_short=self.name_short,
                           individual_plot=self.individual_plot,
                           overwrite=False)
        if projection_out is None or geotransform_out is None:
            print('Either `projection_out` is `None` or `geotransform_out` is '
                  '`None` (or both are). Retrieving projection and '
                  'geotransform information by loading `self.fname_in` via '
                  'GDAL. Be sure this is appropriate for the data you are '
                  'trying to write.\n')
            img_ds = self._read_envi_gdal()
        if projection_out is None:
            projection_out = img_ds.GetProjection()
        if geotransform_out is None and metadata is None:
            geotransform_out = img_ds.GetGeoTransform()
        elif geotransform_out is None and metadata is not None:
            map_set = metadata['map info']
            ul_x_utm = self.tools.get_meta_set(map_set, 3)
            ul_y_utm = self.tools.get_meta_set(map_set, 4)
            size_x_m = self.tools.get_meta_set(map_set, 5)
            size_y_m = self.tools.get_meta_set(map_set, 6)
            # Note the last pixel size must be negative to begin at upper left
            geotransform_out = [ul_x_utm, size_x_m, 0.0, ul_y_utm, 0.0,
                                -size_y_m]

        drv = gdal.GetDriverByName('GTiff')
        drv.Register()
        if len(array.shape) == 3:
            ysize, xsize, bands = array.shape
        else:
            bands = 1
        if bands >= 3:
            tif_out = drv.Create(fname_tif, xsize, ysize, 3,
                                 gdal.GDT_Float32)

            msg = ('GDAL driver was unable to successfully create the empty '
                   'geotiff; check to be sure the correct filename is being '
                   'passed: {0}\n'.format(fname_tif))
            assert tif_out is not None, msg
            tif_out.SetProjection(projection_out)
            tif_out.SetGeoTransform(geotransform_out)

            band_b = self.tools.get_band(460)
            band_g = self.tools.get_band(550)
            band_r = self.tools.get_band(640)
            band_list = [band_r, band_g, band_b]  # backwards for RGB display
#            array_img = None
            for idx, band in enumerate(band_list):
                array_band = array[:, :, band-1]
                if len(array_band.shape) > 2:
                    array_band = array_band.reshape((array_band.shape[0],
                                                     array_band.shape[1]))
                band_out = tif_out.GetRasterBand(idx + 1)
                if np.ma.is_masked(array_band):
                    array_band[array_band.mask] = 0
                    band_out.SetNoDataValue(0)
                band_out.WriteArray(array_band)  # must flip

#                if array_img is None:
#                    array_img = array_band
#                else:
#                    array_img = np.dstack((array_img, array_band))  # stacks bands
                band_out = None
            self.show_img(array, band_r=band_r, band_g=band_g,
                          band_b=band_b, inline=inline)

        else:
            if len(array.shape) == 3:
                array = np.reshape(array, array.shape[:2])
            ysize, xsize = array.shape
            tif_out = drv.Create(fname_tif, xsize, ysize, 1, gdal.GDT_Float32)
            tif_out.SetProjection(projection_out)
            tif_out.SetGeoTransform(geotransform_out)
            band_out = tif_out.GetRasterBand(1)
            if np.ma.is_masked(array):
                array[array.mask] = 0
                band_out.SetNoDataValue(0)
            band_out.WriteArray(array)
            band_out = None
            self.show_img(array, inline=inline)
#        array_img = None

        tif_out.FlushCache()
        drv = None
        tif_out = None


class hstools(object):
    '''
    Basic tools for manipulating Spyfiles and accessing their metadata.

    Parameters:
        spyfile (``SpyFile`` object): The datacube being accessed and/or
            manipulated.
    '''
    def __init__(self, spyfile, ):
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
            metadata (``dict``): dictionary of the metadata
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The datacube being
                accessed and/or manipulated.
        '''
        if spyfile is None:
            spyfile = self.spyfile
        if metadata is None:
            metadata = spyfile.metadata
        meta_bands = {}
        if 'band names' not in metadata.keys():
            try:
                for key, val in enumerate(metadata['wavelength']):
                    meta_bands[key+1] = float(val)
                metadata['band names'] = list(meta_bands.keys())
            except KeyError as err:  # 'wavelength' is not a metadata key
                pass  # meta_bands will just be empty
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
                try:
                    meta_bands[int(band_names[idx])] = float(wl_names[idx])
                except ValueError:
                    meta_bands[band_names[idx]] = float(wl_names[idx])
        self.meta_bands = meta_bands

    def clean_md_sets(self, metadata=None):
        '''

        Parameters:
            metadata (``dict``, optional): Metadata dictionary to clean

        Returns:
            ``dict``:
                **metadata_out** (``dict``) -- Cleaned metadata dictionary.
        '''
        if metadata is None:
            metadata = self.spyfile.metadata
        metadata_out = metadata.copy()
        for key, value in metadata.items():
            if isinstance(value, list):
                set_str = '{' +', '.join(str(x) for x in value) + '}'
                metadata_out[key] = set_str
        return metadata_out

    def del_meta_item(self, metadata, key):
        '''
        Deletes metadata item from SpyFile object.

        Parameters:
            metadata (``dict``): dictionary of the metadata
            key (``str``): dictionary key to delete

        Returns:
            ``dict``:
                **metadata** (``dict``) -- Dictionary containing the modified
                metadata.
        '''
        msg = ('Please be sure to base a metadata dictionary.')
        assert isinstance(metadata, dict), msg
        try:
#            del metadata[key]
            _ = metadata.pop(key, None)
#            val = None
        except KeyError:
            print('{0} not a valid key in input dictionary.'.format(key))
        return metadata

    def get_band(self, target_wl, spyfile=None):
        '''
        Finds the band number of the closest target wavelength.

        Parameters:
            target_wl (``int`` or ``float``): the target wavelength to retrieve the
                band number for (required).
            spyfile (``SpyFile`` object, optional): The datacube being accessed and/or
                manipulated; if ``None``, uses ``hstools.spyfile`` (default:
                ``None``).

        Returns:
            ``int``:
                **key_band** (``int``) -- band number of the closest target
                wavelength (``target_wl``).

        Example:
            >>> hstools.get_band(703, spyfile)
            151
        '''
        if spyfile is None:
            spyfile = self.spyfile
        else:
            self.load_spyfile(spyfile)

        val_target = min(list(self.meta_bands.values()),
                         key=lambda x: abs(x-target_wl))
        key_band = list(self.meta_bands.keys())[sorted(list(
                self.meta_bands.values())).index(val_target)]
#        key_wavelength = sorted(list(self.meta_bands.values()))[key_band-1]
        return key_band

    def get_wavelength(self, target_band, spyfile=None):
        '''
        Returns actual wavelength of the closest target band.

        Parameters:
            target_band (``int`` or ``float``): the target band to retrieve
                wavelength number for (required).
            spyfile (``SpyFile`` object, optional): The datacube being accessed and/or
                manipulated; if ``None``, uses ``hstools.spyfile`` (default:
                ``None``).

        Returns:
            ``float``:
                **key_wavelength** (``float``) -- wavelength of the closest
                target band (``target_band``).

        Example:
            >>> hstools.get_wavelength(151, spyfile)
            702.52
        '''
        if spyfile is None:
            spyfile = self.spyfile
        else:
            self.load_spyfile(spyfile)

        val_target = min(list(self.meta_bands.keys()),
                         key=lambda x: abs(x-target_band))
        key_wavelength = list(self.meta_bands.values())[sorted(list(
                self.meta_bands.keys())).index(val_target)]
#        key_wavelength = sorted(list(self.meta_bands.values()))[key_band-1]
        return key_wavelength

    def get_center_wl(self, wl_list, spyfile=None, wls=True):
        '''
        Gets band numbers and mean wavelength from all wavelengths (or bands)
        in ``wl_list``.

        Parameters:
            wl_list (``list``): the list of bands to get information for
                (required).
            spyfile (``SpyFile`` object): The datacube being accessed and/or
                manipulated; if ``None``, uses ``hstools.spyfile`` (default:
                ``None``).
            wls (``bool``): whether wavelengths are passed in ``wl_list`` or if
                bands are passed in ``wl_list`` (default: ``True`` - wavelenghts
                passed).

        Returns:
            2-element ``tuple`` containing

            - **bands** (``list``): the list of bands (band number) corresponding
              to ``wl_list``.
            - **wls_mean** (``float``): the mean wavelength from ``wl_list``.
        '''
        msg = ('"wl_list" must be a list.')
        assert isinstance(wl_list, list), msg

        if spyfile is None:
            spyfile = self.spyfile
        else:
            self.load_spyfile(spyfile)

        bands = []
        wavelengths = []
        if wls is False:
            for band in wl_list:
                wl_i = self.get_wavelength(band)
                band_i = self.get_band(wl_i)
                bands.append(band_i)
                wavelengths.append(wl_i)
            wls_mean = np.mean(wavelengths)
        else:
            for wl in wl_list:
                band_i = self.get_band(wl)
                wl_i = self.get_wavelength(band_i)
                bands.append(band_i)
                wavelengths.append(wl_i)
            wls_mean = np.mean(wavelengths)

        return bands, wls_mean

    def get_band_index(self, band_num):
        '''
        Subtracts 1 from ``band_num`` and returns the band index(es).

        Parameters:
            band_num (``int`` or ``list``): the target band number(s) to retrieve
            the band index for (required).

        Returns:
            ``int`` or ``list``:
                **band_idx** (``int``) -- band index of the passed band number
                (``band_num``).

        Example:
            >>> hstools.get_band_index([4, 43, 111])
            [3, 42, 110]
        '''
        if isinstance(band_num, list):
            band_num = np.array(band_num)
            band_idx = list(band_num - 1)
        else:
            band_idx = band_num - 1
        return band_idx

    def get_spectral_mean(self, band_list, spyfile=None):
        '''
        Gets the spectral mean of a datacube from a list of bands

        Parameters:
            band_list (``list``): the list of bands to calculate the spectral
                mean for on the datacube (required).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The datacube being
                accessed and/or manipulated; if ``None``, uses ``hstools.spyfile``
                (default: ``None``).

        Returns:
            ``numpy.array`` or ``pandas.DataFrame``:
                **array_mean** (``numpy.array`` or ``pandas.DataFrame``): The
                mean reflectance from ``spyfile`` for the bands in ``band_list``.
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
        elif isinstance(spyfile, pd.DataFrame):
            array = spyfile.copy()

        band_idx = self.get_band_index(band_list)
        if isinstance(array, np.ndarray) and len(array.shape) == 3:
            array_mean = np.mean(array[:, :, band_idx], axis=2)
        elif isinstance(array, np.ndarray) and len(array.shape) == 2:
            array_mean = np.mean(array[:, band_idx], axis=1)
        elif isinstance(array, np.ndarray):
            array_mean = array
        elif isinstance(array, pd.DataFrame):
            str_name = str(band_list)
            array_mean = array[band_list].mean(axis=1).rename(str_name)
        else:
            msg = ('{0} type not supported.\n'.format(type(spyfile)))
            raise TypeError(msg)
        return array_mean

    def get_band_num(self, band_idx):
        '''
        Adds 1 to ``band_idx`` and returns the band number(s).

        Parameters:
            band_idx (``int`` or ``list``): the target band index(es) to retrive
                the band number for (required).

        Returns:
            ``int`` or ``list``:
                **band_num** (``int`` or ``list``): band number of the passed
                band index (``band_idx``).

        Example:
            >>> hstools.get_band_num([4, 43, 111])
            [5, 44, 112]
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
            range_wl (``list``): the minimum and maximum wavelength to consider;
                values should be ``int`` or ``float``.
            index (bool): Indicates whether to return the band number (min=1)
                or to return index number (min=0) (default: True)

        Returns:
            ``list``:
                **band_list** (``list``): A list of all bands (either index or
                number, depending on how ``index`` is set) between a range in
                wavelength values.
        '''
        msg = ('"range_wl" must be a `list` or `tuple`.')
        assert isinstance(range_wl, list) or isinstance(range_wl, tuple), msg
        # could also just take the min and max as long as it has at least 2..
        msg = ('"range_wl" must have exactly two items.')
        assert len(range_wl) == 2, msg

        band_min = self.get_band(min(range_wl))  # gets closest band
        band_max = self.get_band(max(range_wl))
        wl_min = self.get_wavelength(band_min)
        wl_max = self.get_wavelength(band_max)
        if wl_min < range_wl[0]:  # ensures its actually within the range
            band_min += 1
            wl_min = self.get_wavelength(band_min)
        if wl_max > range_wl[1]:
            band_max -= 1
            wl_max = self.get_wavelength(band_max)
        if index is True:
            band_min = self.get_band_index(band_min)
            band_max = self.get_band_index(band_max)
        band_list = [x for x in range(band_min, band_max+1)]
        return band_list

    def get_meta_set(self, meta_set, idx=None):
        '''
        Reads metadata "set" (i.e., string representation of a Python set;
        common in .hdr files), taking care to remove leading and trailing
        spaces.

        Parameters:
            meta_set (``str``): the string representation of the metadata set
            idx (``int``): index to be read; if ``None``, the whole list is
                returned (default: ``None``).

        Returns:
            ``list`` or ``str``:
                **metadata_list** (``list`` or ``str``): List of metadata set items
                (as ``str``), or if idx is not ``None``, the item in the position
                described by ``idx``.
        '''
        if isinstance(meta_set, str):
            meta_set_list = meta_set[1:-1].split(",")
        else:
            meta_set_list = meta_set.copy()
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

    def get_UTM(self, pix_e_ul, pix_n_ul, utm_x, utm_y, size_x, size_y):
        '''
        Calculates the new UTM coordinate of cropped plot to modify the
        "map info" tag of the .hdr file.

        Parameters:
            pix_e_ul (``int``): upper left column (easting) where image cropping
                begins.
            pix_n_ul (``int``): upper left row (northing) where image cropping
                begins.
            utm_x (``float``): UTM easting coordinates (meters) of the original
                image (from the upper left).
            utm_y (``float``): UTM northing coordinates (meters) of the original
                image (from the upper left).
            size_x (``float``): Ground resolved distance of the image pixels in
                the x (easting) direction (meters).
            size_y (``float``): Ground resolved distance of the image pixels in
                the y (northing) direction (meters).

        Returns:
            2-element ``tuple`` containing

            - **utm_x_new** (``float``): The modified UTM x coordinate (easting)
              of cropped plot.
            - **utm_y_new** (``float``): The modified UTM y coordinate (northing)
              of cropped plot.
        '''
        utm_x_new = utm_x + ((pix_e_ul + 1) * size_x)
        utm_y_new = utm_y - ((pix_n_ul + 1) * size_y)
        return utm_x_new, utm_y_new

    def load_spyfile(self, spyfile):
        '''
        Loads a ``SpyFile`` (Spectral Python object) for data access and/or
        manipulation by the ``hstools`` class.

        Parameters:
            spyfile (``SpyFile`` object): The datacube being accessed and/or
                manipulated.
        '''
        self.spyfile = spyfile
        self._get_meta_bands(spyfile)

    def mask_array(self, array, metadata, thresh=None, percentile=None,
                   side='lower'):
        '''
        Creates a masked numpy array based on a threshold value. If ``array`` is
        already a masked array, that mask is maintained and the new mask(s) is/
        are added to the original mask.

        Parameters:
            array (``numpy.ndarray``): The data array to mask.
            thresh (``float`` or ``list``): The value for which to base the
                threshold; if ``thresh`` is ``list`` and ``side`` is ``None``, then
                all values in ``thresh`` will be masked; if ``thresh`` is ``list``
                and ``side`` is not ``None``, then only the first value in the
                list will be considered for thresholding (default: ``None``).
            percentile (``float``): The percentile of pixels to mask; if
                ``percentile``=95 and ``side``='lower', the lowest 95% of pixels
                will be masked prior to calculating the mean spectra across
                pixels (default: ``None``; range: 0-100).
            side (``str``): The side of the threshold for which to apply the
                mask. Must be either 'lower', 'upper', or ``None``; if 'lower',
                everything below the threshold will be masked; if ``None``, only
                the values that exactly match the threshol will be masked
                (default: 'lower').

        Returns:
            2-element ``tuple`` containing

            - **array_mask** (``numpy.ndarray``): The masked ``numpy.ndarray``
              based on the passed threshold and/or percentile value.
            - **metadata** (``dict``): The modified metadata.
        '''
        if isinstance(array, np.ma.core.MaskedArray):
            array_m = array.compressed()  # allows for accurate percentile calc
        else:
            array_m = np.ma.masked_array(array, mask=False)
            array_m = array_m.compressed()

        if thresh is None and percentile is None:
            return array, metadata
        if isinstance(thresh, np.ndarray):
            thresh = list(thresh)
        if isinstance(thresh, list) and side is not None:
            thresh = thresh[0]

        if percentile is not None:
            array_pctl = np.nanpercentile(array_m, percentile)
            if side == 'lower':
                mask_array_p = np.ma.masked_less_equal(array, array_pctl)
            elif side == 'upper':
                mask_array_p = np.ma.masked_greater(array, array_pctl)
            elif side is None:
                mask_array_p = np.ma.masked_equal(array, array_pctl)
        else:
            mask_array_p = np.ma.masked_less(array, np.nanmin(array)-1e-6)
        if thresh is not None:
            if side == 'lower':
                mask_array_t = np.ma.masked_less_equal(array, thresh)
            elif side == 'upper':
                mask_array_t = np.ma.masked_greater(array, thresh)
            elif side is None and isinstance(thresh, list):
                mask_array_t = np.ma.MaskedArray(array, np.in1d(array, thresh))
            else:  # side is None; thresh is float or int
                mask_array_t = np.ma.masked_equal(array, thresh)
        else:
            mask_array_t = np.ma.masked_less(array, np.nanmin(array)-1e-6)

        mask_combine = np.logical_or(mask_array_p.mask, mask_array_t.mask)
        try:
            mask_combine = np.logical_or(mask_combine, array_m.mask)
        except AttributeError as err:
            pass  # array_m does not have a mask
        array = np.ma.masked_invalid(array)  # masks out invalid data (e.g., NaN, inf)
        array_mask = np.ma.array(array, mask=mask_combine)
        unmasked_pct = 100 * (array_mask.count() /
                              (array.shape[0]*array.shape[1]))
#        print('Proportion unmasked pixels: {0:.2f}%'.format(unmasked_pct))

        if side is None:
            side_str = 'equal'
        else:
            side_str = side
        hist_str = (" -> hs_process.mask_array[<"
                    "label: 'thresh?' value:{0}; "
                    "label: 'percentile?' value:{1}; "
                    "label: 'side?' value:{2}; "
                    "label: 'unmasked_pct?' value:{3}>]"
                    "".format(thresh, percentile, side_str, unmasked_pct))
        metadata['history'] += hist_str
        return array_mask, metadata

    def mask_datacube(self, spyfile, mask):
        '''
        DO NOT USE; USE mean_datacube() INSTEAD AND PASS A MASK.


        Applies ``mask`` to ``spyfile``, then returns the datcube (as a np.array)
        and the mean spectra

        Parameters:
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The hyperspectral
                datacube to mask.
            mask (``numpy.ndarray``): the mask to apply to ``spyfile``; if ``mask``
                does not have similar dimensions to ``spyfile``, the first band
                (i.e., first two dimensions) of ``mask`` will be repeated n times
                to match the number of bands of ``spyfile``.
        '''
        if isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = self.spyfile.load()
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()

        if isinstance(mask, np.ma.masked_array):
            mask = mask.mask
        if mask.shape != spyfile.shape:
            if len(mask.shape) == 3:
                mask_2d = np.reshape(mask, mask.shape[:2])
            else:
                mask_2d = mask.copy()
            mask = np.empty(spyfile.shape)
            for band in range(spyfile.nbands):
                mask[:, :, band] = mask_2d

        datacube_masked = np.ma.masked_array(array, mask=mask)
        spec_mean = np.nanmean(datacube_masked, axis=(0, 1))
        spec_std = np.nanstd(datacube_masked, axis=(0, 1))
        spec_mean = pd.Series(spec_mean)
        spec_std = pd.Series(spec_std)
        return spec_mean, spec_std, datacube_masked

    def mean_datacube(self, spyfile, mask=None):
        '''
        Calculates the mean spectra for a datcube; if ``mask`` is passed (as a
        np.array), then the mask is applied to ``spyfile`` prior to computing
        the mean spectra.

        Parameters:
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The hyperspectral
                datacube to mask.
            mask (``numpy.ndarray``): the mask to apply to ``spyfile``; if ``mask``
                does not have similar dimensions to ``spyfile``, the first band
                (i.e., first two dimensions) of ``mask`` will be repeated n times
                to match the number of bands of ``spyfile``.

        Returns:
            3-element ``tuple`` containing

            - **spec_mean** (``SpyFile.SpyFile`` object): The mean spectra from
              the input datacube.
            - **spec_std** (``SpyFile.SpyFile`` object): The standard deviation
              of the spectra from the input datacube.
            - **datacube_masked** (``numpy.ndarray``): The masked
              ``numpy.ndarray``; if ``mask`` = ``None``, ``datacube_masked`` is
              identical to the ``SpyFile`` data array.
        '''
        if isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = self.spyfile.load()
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()

        if isinstance(mask, np.ma.masked_array):
            mask = mask.mask
        if mask is not None:
            if mask.shape != spyfile.shape:
                if len(mask.shape) == 3:
                    mask_2d = np.reshape(mask, mask.shape[:2])
                else:
                    mask_2d = mask.copy()
                mask = np.empty(spyfile.shape)
                for band in range(spyfile.nbands):
                    mask[:, :, band] = mask_2d

        datacube_masked = np.ma.masked_array(array, mask=mask)
        spec_mean = np.nanmean(datacube_masked, axis=(0, 1))
        spec_std = np.nanstd(datacube_masked, axis=(0, 1))
        spec_mean = pd.Series(spec_mean)
        spec_std = pd.Series(spec_std)
        return spec_mean, spec_std, datacube_masked

    def mask_shadow(self, shadow_pctl=20, show_histogram=False,
                    spyfile=None):
        '''
        Creates a ``numpy.mask`` of all pixels that are likely shadow pixels.

        Parameters:
            shadow_pctl (``int``): the percentile of pixels in the image to mask
                (default: 20).
            show_histogram (``bool``):
            spyfile (``SpyFile.SpyFile`` object):

        Returns:
            2-element ``tuple`` containing

            - **array_noshadow.mask** (``numpy.mask``): The mask indicating all
              pixels that are likely shadow pixels.
            - **metadata** (``dict``): The modified metadata.
        '''
        if spyfile is None:
            spyfile = self.spyfile
            array = self.spyfile.load()
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = self.spyfile.load()
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()

        array_energy = np.mean(array, axis=2)
        array_noshadow, metadata = self.mask_array(
                array_energy, self.spyfile.metadata, percentile=shadow_pctl,
                side='lower', show_histogram=show_histogram)
        return array_noshadow.mask, metadata

    def modify_meta_set(self, meta_set, idx, value):
        '''
        Modifies metadata "set" (i.e., string representation of a Python set;
        common in .hdr files) by converting string to list, then adjusts the
        value of an item by its index.

        Parameters:
            meta_set (``str``): the string representation of the metadata set
            idx (``int``): index to be modified; if ``None``, the whole meta_set is
                returned (default: ``None``).
            value (``float``, ``int``, or ``str``): value to replace at idx

        Returns:
            ``str``:
                **set_str** (``str``): Modified metadata set string.
        '''
        metadata_list = self.get_meta_set(meta_set, idx=None)
        metadata_list[idx] = str(value)
        set_str = '{' + ', '.join(str(x) for x in metadata_list) + '}'
        return set_str
