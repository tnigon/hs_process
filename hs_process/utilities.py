# -*- coding: utf-8 -*-
import ast
from matplotlib import pyplot as plt
import numpy as np
import os
from osgeo import gdal
from osgeo import gdalconst
import pandas as pd
import re
import seaborn as sns
import spectral.io.envi as envi
import spectral.io.spyfile as SpyFile
import sys
import sysconfig
import warnings

plt_style = 'seaborn-whitegrid'
plt.style.use(plt_style)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('text', usetex=False)


class _dotdict(dict):
    '''dot.notation access to dictionary attributes'''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class defaults(object):
    '''
    Class containing default values and/or settings for various
    ``hs_process`` tools/functions.
    '''
    def __init__(self):
        self.crop_defaults = _dotdict({
                'directory': None,
                'name_short': None,
                'name_long': None,
                'ext': 'bip',
                'plot_id_ref': None,
                'pix_e_ul': 0,
                'pix_n_ul': 0,
                'alley_size_e_m': None,
                'alley_size_n_m': None,
                'alley_size_e_pix': None,  # set to `None` because should be set
                'alley_size_n_pix': None,  # intentionally
                'buf_e_m': None,
                'buf_n_m': None,
                'buf_e_pix': None,
                'buf_n_pix': None,
                'crop_e_m': None,
                'crop_n_m': None,
                'crop_e_pix': None,
                'crop_n_pix': None,
                'gdf_shft_e_m': None,
                'gdf_shft_n_m': None,
                'gdf_shft_e_pix': None,
                'gdf_shft_n_pix': None,
                'n_plots': None})
        '''
        Default values for performing spatial cropping on images. ``crop_defaults``
        is referenced by the ``spatial_mod.crop_single()`` function to get default
        values if various user-parameters are not passed or are left to ``None``.
        In this way, ``defaults.crop_defaults`` can be modified once by the user to
        avoid having to pass the same parameter(s) repeatedly if executing
        ``spatial_mod.crop_single()`` many times, such as in a for loop.

        Attributes:
            crop_defaults.directory (``str``): File directory of the input image
                to be cropped (default: ``None``).
            crop_defaults.name_short (``str``): Part of the datacube name that is
                generally not repeated across many datacubes captured at the
                same time. In the ``name_long`` example above,
                ``name_short`` = "plot_101_pika_gige_2". The part of the
                filename that is ``name_short`` should end with a dash (but
                should not include that dash as it belongs to ``name_long``;
                default: ``None``).
            crop_defaults.name_long (``str``): Part of the datacube name that tends
                to be long and is repeated across many datacubes captured at
                the same time. This is an artifact of Resonon/Spectronon
                software, and may be desireable to shorten and/or make more
                informative. For example, a datacube may have the following name:
                *"plot_101_pika_gige_2-Radiance From Raw Data-Georectify Airborne Datacube-Reflectance from Radiance Data and Measured Reference Spectrum.bip"*
                and another datacube captured in the same campaign may be named:
                *"plot_102_pika_gige_1-Radiance From Raw Data-Georectify Airborne Datacube-Reflectance from Radiance Data and Measured Reference Spectrum.bip"*
                ``name_long`` should refer to everything after the first dash
                (including the first dash) up to the file extension (".bip"):
                ``name_long`` = *"-Radiance From Raw Data-Georectify Airborne Datacube-Reflectance from Radiance Data and Measured Reference Spectrum"*
                (default: ``None``).
            crop_defaults.ext (``str``): File extension to save the cropped image
                (default: 'bip').
            crop_defaults.pix_e_ul (``int``): upper left pixel column (easting) to
                begin cropping (default: 0).
            crop_defaults.pix_n_ul (``int``): upper left pixel row (northing) to
                begin cropping (default: 0).
            crop_defaults.buf_e_pix (``int``): The buffer distance in the easting
                direction (in pixel units) to be applied after calculating the
                original crop area (default: 0).
            crop_defaults.buf_n_pix (``int``): The buffer distance in the northing
                direction (in pixel units) to be applied after calculating the
                original crop area (default: 0).
            crop_defaults.buf_e_m (``float``): The buffer distance in the easting
                direction (in map units; e.g., meters) to be applied after
                calculating the original crop area; the buffer is considered
                after ``crop_X_m``/``crop_X_pix``. A positive value will
                reduce the size of ``crop_X_m``/``crop_X_pix``, and a
                negative value will increase it (default: ``None``).
            crop_defaults.buf_n_m (``float``): The buffer distance in the northing
                direction (in map units; e.g., meters) to be applied after
                calculating the original crop area; the buffer is considered
                after ``crop_X_m``/``crop_X_pix``. A positive value will
                reduce the size of ``crop_X_m``/``crop_X_pix``, and a
                negative value will increase it (default: ``None``).
            crop_defaults.crop_e_pix (``int``): number of pixels in each row in the
                cropped image (default: 90).
            crop_defaults.crop_n_pix (``int``): number of pixels in each column in
                the cropped image (default: 120).
            crop_defaults.crop_e_m (``float``): length of each row (easting
                direction) of the cropped image in map units (e.g., meters;
                default: ``None``).
            crop_defaults.crop_n_m (``float``): length of each column (northing
                direction) of the cropped image in map units (e.g., meters;
                default: ``None``).
            crop_defaults.plot_id_ref (``int``): the plot ID of the area to be cropped
                (default: ``None``).
        '''

        self.envi_write = _dotdict({'dtype': np.float32,
                                    'force': False,
                                    'ext': '',
                                    'interleave': 'bip',
                                    'byteorder': 0})
        '''
        Attributes for writing ENVI datacubes to file, following the convention of
        the `Spectral Python`_ `envi.save_image()`_ parameter options for writing
        an ENVI datacube to file.

        Attributes:
            envi_write.dtype (``numpy.dtype`` or ``str``): The data type with which
                to store the image. For example, to store the image in 16-bit
                unsigned integer format, the argument could be any of
                ``numpy.uint16``, ``'u2'``, ``'uint16'``, or ``'H'`` (default:
                ``np.float32``).
            envi_write.force (``bool``): If ``hdr_file`` or its associated image
                file exist, ``force=True`` will overwrite the files; otherwise, an
                exception will be raised if either file exists (default:
                ``False``).
            envi_write.ext (``str``): The extension to use for saving the image
                file; if not specified, a default extension is determined based on
                the ``interleave``. For example, if ``interleave``='bip', then
                ``ext`` is set to 'bip' as well. If ``ext`` is an empty string, the
                image file will have the same name as the .hdr, but without the
                '.hdr' extension (default: ``None``).
            envi_write.interleave (``str``): The band interleave format to use for
                writing the file; ``interleave`` should be one of 'bil', 'bip', or
                'bsq' (default: 'bip').
            envi_write.byteorder (``int`` or ``str``): Specifies the byte order
                (endian-ness) of the data as written to disk. For little endian,
                this value should be either 0 or 'little'. For big endian, it
                should be either 1 or 'big'. If not specified, native byte order
                will be used (default: ``None``).

        .. _Spectral Python: http://www.spectralpython.net/
        .. _envi.save_image(): http://www.spectralpython.net/class_func_ref.html#spectral.io.envi.save_image
        '''

        self.spat_crop_cols = _dotdict({
                'directory': 'directory',
                'fname': 'fname',
                'name_short': 'name_short',
                'name_long': 'name_long',
                'ext': 'ext',
                'plot_id_ref': 'plot_id_ref',
                'pix_e_ul': 'pix_e_ul',
                'pix_n_ul': 'pix_n_ul',
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
                'gdf_shft_e_m': 'gdf_shft_e_m',
                'gdf_shft_n_m': 'gdf_shft_n_m',
                'gdf_shft_e_pix': 'gdf_shft_e_pix',
                'gdf_shft_n_pix': 'gdf_shft_n_pix',
                'n_plots_x': 'n_plots_x',
                'n_plots_y': 'n_plots_y',
                'n_plots': 'n_plots'})
        '''
        Default column names for performing batch spatial cropping on
        images. Useful when batch processing images via `batch.spatial_crop()`_.
        ``batch.spatial_crop()`` takes a parameter ``fname_sheet``, which can be a
        filename to a spreadsheet or a ``pandas.DataFrame``.
        ``defaults.spat_crop_cols`` should be modified if the column names in
        ``fname_sheet`` are different than what is expected (see documentation for
        `batch.spatial_crop()`_ to know the expected column names).

        Attributes:
            spat_crop_cols.directory (``str``): column name for input directory
                (default: 'directory').
            spat_crop_cols.fname (``str``): column name for input fname
                (default: 'fname').
            spat_crop_cols.name_short (``str``): column name for input image's
                ``name_short`` (default: 'name_short').
            spat_crop_cols.name_long (``str``): column name for input image's
                ``name_long`` (default: 'name_long').
            spat_crop_cols.ext (``str``): column name for file extension of input
                image (default: 'ext').
            spat_crop_cols.pix_e_ul (``str``): column name for ``pix_e_ul``
                (default: 'pix_e_ul').
            spat_crop_cols.pix_n_ul (``str``): column name for ``pix_n_ul``
                (default: 'pix_n_ul').
            spat_crop_cols.alley_size_e_pix (``str``): column name for
                ``alley_size_e_pix`` (default: 'alley_size_e_pix').
            spat_crop_cols.alley_size_n_pix (``str``): column name for
                ``alley_size_n_pix`` (default: 'alley_size_n_pix').
            spat_crop_cols.alley_size_e_m (``str``): column name for
                ``alley_size_e_m`` (default: 'alley_size_e_m').
            spat_crop_cols.alley_size_n_m (``str``): column name for
                ``alley_size_n_m`` (default: 'alley_size_n_m').
            spat_crop_cols.buf_e_pix (``str``): column name for ``buf_e_pix``
                (default: 'buf_e_pix').
            spat_crop_cols.buf_n_pix (``str``): column name for ``buf_n_pix``
                (default: 'buf_n_pix').
            spat_crop_cols.buf_e_m (``str``): column name for ``buf_e_m`` (default:
                'buf_e_m').
            spat_crop_cols.buf_n_m (``str``): column name for ``buf_n_m`` (default:
                'buf_n_m').
            spat_crop_cols.crop_e_pix (``str``): column name for ``crop_e_pix``
                (default: 'crop_e_pix').
            spat_crop_cols.crop_n_pix (``str``): column name for ``crop_n_pix``
                (default: 'crop_n_pix').
            spat_crop_cols.crop_e_m (``str``): column name for ``crop_e_m``
                (default: 'crop_e_m').
            spat_crop_cols.crop_n_m (``str``): column name for ``crop_n_m``
                (default: 'crop_n_m').
            spat_crop_cols.plot_id_ref (``str``): column name for ``plot_id``
                (default: 'crop_n_pix').
            spat_crop_cols.n_plots_x (``str``): column name for ``n_plots_x``
                (default: 'n_plots_x').
            spat_crop_cols.n_plots_y (``str``): column name for ``n_plots_y``
                (default: 'n_plots_y').
            spat_crop_cols.n_plots (``str``): column name for ``n_plots``
                (default: 'n_plots').

        .. _batch.spatial_crop(): hs_process.batch.html#hs_process.batch.spatial_crop
        '''
#    dtype = np.float32
#    force = False
#    ext = ''
#    interleave = 'bip'
#    byteorder = 0


class hsio(object):
    '''
    Class for reading and writing hyperspectral data files, as well as
    accessing, interpreting, and modifying its associated metadata. With a
    hyperspectral data file loaded via ``hsio``, there is simple
    functionality to display the datacube image as a multi-band render, as
    well as for saving a datacube as a 3-band geotiff. ``hsio`` relies
    heavily on the `Spectral Python`_ package.

    .. _Spectral Python: http://www.spectralpython.net/
    '''
    def __init__(self, fname_in=None, name_long=None, name_plot=None,
                 name_short=None, str_plot='plot_', individual_plot=False,
                 fname_hdr_spec=None):
        '''
        Parameters:
            fname_in (``str``, optional): The filename of the image datacube to
                be read in initially.
            name_long (``str``, optional): Part of the datacube name that tends
                to be long and is repeated across many datacubes captured at
                the same time. This is an artifact of Resonon/Spectronon
                software, and may be desireable to shorten and/or make more
                informative. For example, a datacube may have the following
                name:
                "plot_101_pika_gige_2-Radiance From Raw Data-Georectify Airborne Datacube-Reflectance from Radiance Data and Measured Reference Spectrum.bip"
                and another datacube captured in the same campaign may be
                named:
                "plot_102_pika_gige_1-Radiance From Raw Data-Georectify Airborne Datacube-Reflectance from Radiance Data and Measured Reference Spectrum.bip"
                ``name_long`` should refer to everything after the first dash
                (including the first dash) up to the file extension (".bip"):
                ``name_long`` = "-Radiance From Raw Data-Georectify Airborne Datacube-Reflectance from Radiance Data and Measured Reference Spectrum"
                (default: ``None``).
            name_plot (``str``, optional): Part of the datacube name that
                refers to the plot name and/or ID. It usually includes "plot"
                in its name. In the ``name_long`` example above,
                ``name_plot`` = "plot_101" (default: ``None``).
            name_short (``str``, optional): Part of the datacube name that is
                generally not repeated across many datacubes captured at the
                same time. In the ``name_long`` example above,
                ``name_short`` = "plot_101_pika_gige_2". The part of the
                filename that is ``name_short`` should end with a dash (but
                should not include that dash as it belongs to ``name_long``;
                default: ``None``).
            str_plot (``str``, optional): Part of ``name_plot`` that, when
                removed, should leave a numeric value that can be interpreted
                as an integer data type. In the ``name_plot`` example above, if
                "plot_" is removed from "plot_101", "101" remains and can be
                interpreted as an integer using ``int("101")`` (default:
                "plot_").
            individual_plot (``bool``, optional): Indicates whether the
                datacube (and its filename) is for an individual plot
                (``True``), or for many plots (``False``; default: ``False``).
            fname_hdr_spec (``str``, optional): The ``hsio`` class can also
                read in a "spectrum" (.spec) file (i.e., a hyperspectral file
                without any spatial infomation and only a single spectral
                profile; there is only a single "pixel").
                ``fname_hdr_spec`` refers to the full file path of the spectrum
                file (default: ``None``).
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
            self.read_cube(fname_hdr=self.fname_hdr, overwrite=False,
                           name_long=self.name_long,
                           name_short=self.name_short,
                           name_plot=self.name_plot,
                           individual_plot=individual_plot)

        if self.fname_hdr_spec is not None:
            self.read_spec(self.fname_hdr_spec)

        self.defaults = defaults()

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

    def _parse_fname_plot(self, str_plot):
        '''
        Code for parsing ``name_plot`` (numeric text following ``str_plot``).
        '''
        s = self.name_short
        if str_plot in s and '_pika' in s:
            name_plot = s[s.find(str_plot) + len(str_plot):s.find('_pika')]
        elif str_plot in s and '_pika' not in s:
            if s.find('_', s.find(str_plot) + len(str_plot)) == -1:
                name_plot = s[s.find(str_plot) + len(str_plot):]
            else:  # there is an underscore after plot_id, so cut off name_plot there
                name_plot = s[s.find(str_plot) + len(str_plot):s.find('_')]
        else:
            name_plot = self.name_short.rsplit('_', 1)[1]

        if len(name_plot) > 12:  # then it must have gone wrong
            name_plot = self.name_short.rsplit('_', 1)[1]
        if len(name_plot) == 0:  # then '_pika' must not
            name_plot = self.name_short.rsplit('_', 1)[1]

        try:
            int(name_plot)
        except ValueError:  # give up..
            msg = ('Cannot determine the plot name from the image filename. '
                   'Setting `hsio.name_plot` to `None`. If this image is for '
                   'a particular plot, please set `hsio.name_plot; otherwise, '
                   'ignore this warning.\n')
            warnings.warn(msg, UserWarning)
            name_plot = None
        return name_plot

    def _parse_fname(self, fname_hdr=None, str_plot='plot_', overwrite=True,
                     name_long=None, name_short=None, name_plot=None):
        '''
        Parses the filename for ``name_long`` (text after the first dash,
        inclusive), ``name_short`` (text before the first dash), and
        ``name_plot`` (numeric text following ``str_plot``).

        Parameters:
            fname_hdr (``str``): input filename.
            str_plot (``str``): text to search for that precedes the numeric
                text that describes the plot number.
            overwrite (``bool``): whether the class instances of ``name_long``,
                ``name_short``, and ``name_plot`` should be overwritten based
                on ``fname_in`` (default: ``True``).
        '''
        if fname_hdr is None:
            fname_hdr = self.fname_hdr + '.hdr'
        if os.path.splitext(fname_hdr)[1] == '.hdr':  # modify self.fname_in based on new file
            fname_in = os.path.splitext(fname_hdr)[0]
        else:
            fname_hdr = fname_hdr + '.hdr'
            fname_in = os.path.splitext(fname_hdr)[0]
        self.fname_in = fname_in
        self.fname_hdr = fname_hdr

        self.base_dir = os.path.dirname(fname_in)
#        base_name = os.path.basename(fname_in)
        base_name = os.path.basename(os.path.splitext(fname_in)[0])
        self.base_name = base_name

        if overwrite is True:
            if '-' in base_name:
                self.name_long = base_name[base_name.find(
                        '-', base_name.rfind('_')):]
                self.name_short = base_name[:base_name.find(
                        '-', base_name.rfind('_'))]
            else:
                # if name_long does not have ext, it can be just blank
                self.name_long = ''
                # and name_short can be base_name
                self.name_short = base_name
            self.name_plot = self._parse_fname_plot(str_plot)

        if name_long is not None:
            self.name_long = name_long
        elif overwrite is False and self.name_long is None:
            if '-' in base_name:
                self.name_long = base_name[base_name.find(
                        '-', base_name.rfind('_')):]
            else:
                # if name_long does not have ext, it can be just blank
                self.name_long = ''
        else:
            pass

        if name_short is not None:
            self.name_short = name_short
        elif overwrite is False and self.name_short is None:
            if '-' in base_name:
                self.name_short = base_name[:base_name.find(
                        '-', base_name.rfind('_'))]
            else:
                self.name_short = base_name
        else:
            pass

        if name_plot is not None:
            self.name_plot = name_plot
        elif overwrite is False and self.name_plot is None:
            self.name_plot = self._parse_fname_plot(str_plot)
        else:
            pass

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

    def _get_fname_hdr(self, fname_hdr, ext=None,
                       interleave=None):
        '''
        Checks to be sure .hdr filename follows that of the extension. There
        is an exception for .spec files because they are treated differently
        in Spectronon software.

        Order of operations:

        1. If ext is not set and filename doesn't appear to have an extension,
            use the interleave as the ext
        2. If ext is not set and filename has an extension, use the filename as
            the ext
        3. If ext is set, be sure fname follows; if so, no issues.
        4. Otherwise, if fname does not follow, use ext
        5. Else, use ".bip.hdr"
        '''
        if ext is None:
            ext = self.defaults.envi_write.ext
        if interleave is None:
            interleave = self.defaults.envi_write.interleave

        if os.path.splitext(fname_hdr)[1] == '.hdr':
            file_wo_hdr = os.path.splitext(fname_hdr)[0]
            file_wo_ext = os.path.splitext(file_wo_hdr)[0]
        else:
            file_wo_hdr = fname_hdr
            file_wo_ext = os.path.splitext(file_wo_hdr)[0]

        if (ext is None or ext == '') and os.path.splitext(file_wo_hdr)[1] == '':  # must use interleave
            fname_hdr = file_wo_ext + '.' + interleave + '.hdr'
        elif (ext is None or ext == '') and os.path.splitext(file_wo_hdr)[1] != '':  # use filename
            fname_hdr = file_wo_hdr + '.hdr'
        elif os.path.splitext(file_wo_hdr)[1] == ext:  # both ext and file_wo_hdr are good
            fname_hdr = file_wo_hdr + '.hdr'
        elif os.path.splitext(file_wo_hdr)[1] != ext:  # must use ext unless
            if ext[0] != '.':
                ext = '.' + ext
            fname_hdr = file_wo_ext + ext + '.hdr'
        else:
            fname_hdr = file_wo_ext + '.bip.hdr'
            # if os.path.splitext(fname_hdr)[1] != '.hdr':
            #     fname_hdr = file_wo_hdr + '.hdr'  # we know it includes ext
        return fname_hdr
        # if ext is None or ext == '':
        #     if os.path.splitext(file_wo_ext)[1] == '':  # must use interleave
        #         fname_hdr = file_wo_ext + '.' +\
        #                 interleave + '.hdr'


    def _check_data_size(self, spyfile, func='write_cube', fname_out=None):
        '''
        Ensures there is data present; if not, prints a warning and returns
        ``False``
        '''
        basename = os.path.basename(fname_out)
        if isinstance(spyfile, SpyFile.SpyFile):
            if spyfile.ncols == 0 or spyfile.nrows == 0 or spyfile.nbands == 0:
                print('The size of ``spyfile`` is zero; thus there is nothing '
                      'to write to file and ``{0}()`` is being '
                      'aborted.\nFilename: {1}\n'.format(func, basename))
                return False
            elif np.isnan(spyfile.load()).all():  # CAUTION: May take several seconds to laod spyfile as array
                print('All pixels in ``spyfile`` are null values (NaN); thus '
                      '``{0}()`` is being aborted.\nFilename: {1}\n'
                      ''.format(func, basename))
                return False
        elif isinstance(spyfile, np.ndarray):
            if spyfile.size == 0:
                print('The size of ``spyfile`` is zero; thus there is nothing '
                      'to write to file and ``{0}()`` is being '
                      'aborted.\nFilename: {1}\n'.format(func, basename))
                return False
            elif np.isnan(spyfile).all():
                print('All pixels in ``spyfile`` are null values (NaN); thus '
                      '``{0}()`` is being aborted.\nFilename: {1}\n'
                      ''.format(func, basename))
                return False
        else:
            return True

    def read_cube(self, fname_hdr=None, overwrite=True, name_long=None,
                  name_short=None, name_plot=None, individual_plot=False):
        '''
        Reads in a hyperspectral datacube using the `Spectral Python`_ package.

        Parameters:
            fname_hdr (``str``): filename of datacube to be read (default:
                ``None``).
            overwrite (``bool``): Whether to overwrite any of the previous
                user-passed variables, including ``name_long``, ``name_plot``,
                and ``name_short``. If variables are already set and
                ``overwrite`` is ``False``, they will remain the same. If
                variables are set and ``overwrite`` is ``True``, they will be
                overwritten based on typcial file naming conventions of
                Resonon/Spectronon software. Any of the user-passed variables
                (e.g., ``name_long``, etc.) will overwrite those that were set
                previously whether ``overwrite`` is ``True`` or ``False``
                (default: ``False``).
            name_long (``str``): Spectronon processing appends processing names
                to the filenames; this indicates those processing names that
                are repetitive and can be deleted from the filename following
                processing (default: ``None``).
            name_short (``str``): The base name of the image file (see note
                above about ``name_long``; default: ``None``).
            name_plot (``str``): numeric text that describes the plot number
                (default: ``None``).
            individual_plot (``bool``): Indicates whether image (and its
                filename) is for an individual plot (``True``), or for many
                plots (``False``; default: ``False``).

        Note:
            ``hs_process`` will search for ``name_long``, ``name_plot``, and
            ``name_short`` based on typical file naming behavior of Resonon/
            Spectronon software. If any of these parameters are passed by the
            user, however, that will take precedence over "searching the
            typical file nameing behavior".

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_hdr = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio()  # initialize an instance of the hsio class (note there are no required parameters)

            Load datacube using ``hsio.read_cube``

            >>> io.read_cube(fname_hdr)
            >>> io.spyfile
            Data Source:   'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip'
        	# Rows:            617
        	# Samples:        1300
        	# Bands:           240
        	Interleave:        BIP
        	Quantization:  32 bits
        	Data format:   float32

            Check ``name_long``, ``name_short``, and ``name_plot`` values derived from the filename

            >>> io.name_long
            '-Convert Radiance Cube to Reflectance from Measured Reference Spectrum'

            >>> io.name_plot
            '7'

            >>> io.name_short
            'Wells_rep2_20180628_16h56m_pika_gige_7'

        .. _Spectral Python: http://www.spectralpython.net/
        '''
        if os.path.splitext(fname_hdr)[1] != '.hdr':
            fname_hdr = fname_hdr + '.hdr'
        msg = ('Could not find .hdr file.\nLocation: {0}'.format(fname_hdr))
        assert os.path.isfile(fname_hdr), msg
        self.fname_hdr = fname_hdr
        self.base_dir = os.path.dirname(fname_hdr)

        self.individual_plot = individual_plot
        self._read_envi()

        self._parse_fname(fname_hdr, self.str_plot, overwrite=overwrite,
                          name_long=name_long, name_short=name_short,
                          name_plot=name_plot)

    def read_spec(self, fname_hdr_spec, overwrite=True, name_long=None,
                  name_short=None, name_plot=None):
        '''
        Reads in a hyperspectral spectrum file using the using the `Spectral
        Python`_ package.

        Parameters:
            fname_hdr_spec (``str``): filename of spectra to be read.

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_hdr = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7_plot_611-cube-to-spec-mean.spec.hdr'
            >>> io = hsio()  # initialize an instance of the hsio class (note there are no required parameters)

            Load datacube using ``hsio.read_spec``

            >>> io.read_spec(fname_hdr)
            >>> io.spyfile_spec
            Data Source:   'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7_plot_611-cube-to-spec-mean.spec'
        	# Rows:              1
        	# Samples:           1
        	# Bands:           240
        	Interleave:        BIP
        	Quantization:  32 bits
        	Data format:   float32

            Check ``name_long``, ``name_short``, and ``name_plot`` values derived from the filename

            >>> io.name_long
            '-cube-to-spec-mean'

            >>> io.name_short
            'Wells_rep2_20180628_16h56m_pika_gige_7_plot_611'

            >>> io.name_plot
            '611'

        .. _Spectral Python: http://www.spectralpython.net/
        '''
        if os.path.splitext(fname_hdr_spec)[1] != '.hdr':
            fname_hdr_spec = fname_hdr_spec + '.hdr'
        assert os.path.isfile(fname_hdr_spec), 'Could not find .hdr file.'
        self.fname_hdr_spec = fname_hdr_spec
        self.base_dir_spec = os.path.dirname(fname_hdr_spec)
        self._read_envi(spec=True)

        self._parse_fname(fname_hdr_spec, self.str_plot, overwrite=overwrite,
                          name_long=name_long, name_short=name_short,
                          name_plot=name_plot)
#        basename = os.path.basename(fname_hdr_spec)
#        self.name_short = basename[:basename.find('-', basename.rfind('_'))]
#        self.name_long = basename[basename.find('-', basename.rfind('_')):]
#        self.name_plot = self.name_short.rsplit('_', 1)[1]

    def set_io_defaults(self, dtype=False, force=None, ext=False,
                        interleave=False, byteorder=False):
        '''
        Sets any of the ENVI file writing parameters to ``hsio``; if any
        parameter is left unchanged from its default, it will remain as-is
        (i.e., it will not be set).

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

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> io = hsio()  # initialize an instance of the hsio class

            Check ``defaults.envi_write``

            >>> io.defaults.envi_write
            {'dtype': numpy.float32,
             'force': False,
             'ext': '',
             'interleave': 'bip',
             'byteorder': 0}

            Modify ``force`` parameter and recheck ``defaults.envi_write``

            >>> io.set_io_defaults(force=True)
            >>> io.defaults.envi_write
            {'dtype': numpy.float32,
             'force': True,
             'ext': '',
             'interleave': 'bip',
             'byteorder': 0}
        '''
        if dtype is not False:
            self.defaults.envi_write['dtype'] = dtype
            self.defaults.dtype = dtype
        if force is not None:
            self.defaults.envi_write['force'] = force
            self.defaults.force = force
        if ext is not False:
            self.defaults.envi_write['ext'] = ext
            self.defaults.ext = ext
        if interleave is not False:
            self.defaults.envi_write['interleave'] = interleave
            self.defaults.interleave = interleave
        if byteorder is not False:
            self.defaults.envi_write['byteorder'] = byteorder
            self.defaults.byteorder = byteorder

    def show_img(self, spyfile=None, band_r=120, band_g=76, band_b=32,
                 vmin=None, vmax=None, cmap='viridis', cbar=True,
                 inline=True):
        '''
        Displays a datacube as a 3-band RGB image using `Matplotlib`_.

        Parameters:
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The data cube to
                display; if ``None``, loads from ``self.spyfile`` (default:
                ``None``).
            band_r (``int``): Band to display on the red channel (default: 120)
            band_g (``int``): Band to display on the green channel (default:
                76)
            band_b (``int``): Band to display on the blue channel (default: 32)
            vmin/vmax (``scalar``, optional): The data range that the colormap
                covers. By default, the colormap covers the complete value
                range of the supplied data (default: ``None``).
            cmap (``str``): The Colormap instance or registered colormap name
                used to map scalar data to colors. This parameter is ignored
                for RGB(A) data (default: "viridis").
            cbar (``bool``): Whether to include a colorbar in the image
                (default: ``True``).
            inline (``bool``): If ``True``, displays in the IPython console;
                else displays in a pop-out window (default: ``True``).

        Note:
            The `inline` parameter points to the `hsio.show_img` function, and
            is only expected to work in an IPython console (not intended to be
            used in a normal Python console).

        Example:
            Load ``hsio`` and ``spatial_mod`` modules

            >>> from hs_process import hsio # load hsio
            >>> from hs_process import spatial_mod # load spatial mod

            Load the datacube using ``hsio.read_cube``

            >>> fname_hdr = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio()  # initialize an instance of the hsio class
            >>> io.read_cube(fname_hdr)

            Perform simple spatial cropping via ``spatial_mod.crop_single``

            >>> my_spatial_mod = spatial_mod(io.spyfile)  # initialize spatial_mod instance to crop the image
            >>> array_crop, metadata = my_spatial_mod.crop_single(pix_e_ul=250, pix_n_ul=100, crop_e_m=8, crop_n_m=3)

            Show an RGB render of the cropped image using ``hsio.show_img``

            >>> io.show_img(array_crop)

            .. image:: ../img/utilities/show_img.png

        .. _Matplotlib: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.imshow.html#
        '''
        # if inline is True:
        #     get_ipython().run_line_magic('matplotlib', 'inline')
        # else:
        #     try:
        #         get_ipython().run_line_magic('matplotlib', 'auto')
        #     except ModuleNotFoundError:
        #         pass  # just go with whatever is already set

        plt.style.use('default')

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

        ax = plt.subplot()

        if n_bands >= 3:
            try:
#                plt.imshow(array, (band_r, band_g, band_b))
                fig = ax.imshow(array, (band_r, band_g, band_b))
            except ValueError as err:
#                plt.imshow(array[:, :, [band_r, band_g, band_b]]*5.0)
                fig = ax.imshow(array[:, :, [band_r, band_g, band_b]]*5.0)
#            array_img_out = array_img[:, :, [band_r, band_g, band_b]]
#            array_img_out *= 3.5  # Images are very dark without this

        elif n_bands == 1:
            array = np.squeeze(array)
#            ax.imshow(array, vmin=vmin, vmax=vmax)
            fig = ax.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
#            plt.imshow(array, vmin=vmin, vmax=vmax)
            fig = ax.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap)

        if cbar is True and n_bands ==1:
            my_cbar = plt.colorbar(fig, shrink=0.5, ax=ax)
#        fig.show()
        print('\n')

#lowerBound = 0.25
#upperBound = 0.75
#myMatrix = np.random.rand(100,100)
#
#myMatrix =np.ma.masked_where((lowerBound < myMatrix) &
#                             (myMatrix < upperBound), myMatrix)
#
#
#fig,axs=plt.subplots(2,1)
##Plot without mask
#axs[0].imshow(myMatrix.data)
#
##Default is to apply mask
#axs[1].imshow(myMatrix.mask)
#
#plt.show()

    def write_cube(self, fname_hdr, spyfile, metadata=None, dtype=None,
                   force=None, ext=None, interleave=None, byteorder=None):
        '''
        Wrapper function that accesses the `Spectral Python`_ package to save a
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
                overwrite any existing metadata stored by the ``SpyFile``
                object (default=None).
            dtype (``numpy.dtype`` or ``str``): The data type with which to
                store the image. For example, to store the image in 16-bit
                unsigned integer format, the argument could be any of
                numpy.uint16, 'u2', 'uint16', or 'H' (default=np.float32).
            force (``bool``): If ``hdr_file`` or its associated image file
                exist, ``force=True`` will overwrite the files; otherwise, an
                exception will be raised if either file exists (default=False).
            ext (``None`` or ``str``): The extension to use for saving the
                image file. If not specified or if set to an empty string
                (e.g., ``ext=''``), a default extension is determined using the
                same name as ``fname_hdr``, except without the ".hdr"
                extension. If ``fname_hdr`` is provided without the
                "non-.hdr" extension (e.g., "bip"), then the extension is
                determined from the ``interleave`` parameter. For example, if
                ``interleave``='bip', then ``ext`` is set to 'bip' as well. Use
                of ``ext`` is not recommended; instead, just set
                ``fname_hdr`` with the correct extension or use
                ``interleave`` to set the extension (default: ``None``;
                determined from ``fname_hdr`` or ``interleave``).
            interleave (``str``): The band interleave format to use for writing
                the file; ``interleave`` should be one of 'bil', 'bip', or
                'bsq' (default='bip').
            byteorder (``int`` or ``str``): Specifies the byte order
                (endian-ness) of the data as written to disk. For little
                endian, this value should be either 0 or 'little'. For big
                endian, it should be either 1 or 'big'. If not specified,
                native byte order will be used (default=None).

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

        Example:
            Load ``hsio`` and ``spatial_mod`` modules

            >>> import os
            >>> from hs_process import hsio  # load hsio
            >>> from hs_process import spatial_mod  # load spatial mod
            >>> fname_hdr_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio()  # initialize the hsio class
            >>> io.read_cube(fname_hdr_in)

            Perform simple spatial cropping via ``spatial_mod.crop_single`` to generate a new datacube.

            >>> my_spatial_mod = spatial_mod(io.spyfile)  # initialize spatial_mod instance to crop the image
            >>> array_crop, metadata = my_spatial_mod.crop_single(pix_e_ul=250, pix_n_ul=100, crop_e_m=8, crop_n_m=3)

            Save the datacube using ``hsio.write_cube``

            >>> fname_hdr = r'F:\\nigo0024\Documents\hs_process_demo\hsio\Wells_rep2_20180628_16h56m_pika_gige_7-hsio-write-cube-cropped.bip.hdr'
            >>> os.mkdir(os.path.dirname(fname_hdr))
            >>> io.write_cube(fname_hdr, array_crop, metadata=metadata)
            Saving F:\nigo0024\Documents\hs_process_demo\hsio\Wells_rep2_20180628_16h56m_pika_gige_7-hsio-write-cube-cropped.bip

            Load the datacube into Spectronon for visualization

            .. image:: ../img/utilities/write_cube.png

        .. _Spectral Python: http://www.spectralpython.net/
        '''
        if self._check_data_size(spyfile, func='write_cube',
                                 fname_out=fname_hdr) is False:
            return

        if dtype is None:
#            dtype = self.defaults.envi_write['dtype']
            dtype = self.defaults.envi_write.dtype
        if force is None:
#            force = self.defaults.envi_write['force']
            force = self.defaults.envi_write.force
        if ext is None:
#            ext = self.defaults.envi_write['ext']
            ext = self.defaults.envi_write.ext
        if interleave is None:
#            interleave = self.defaults.envi_write['interleave']
            interleave = self.defaults.envi_write.interleave
        if byteorder is None:
#            byteorder = self.defaults.envi_write['byteorder']
            byteorder = self.defaults.envi_write.byteorder

        if metadata is None and isinstance(spyfile, SpyFile.SpyFile):
            metadata = spyfile.metadata
        elif metadata is None and isinstance(spyfile, np.ndarray):
            raise TypeError('`spyfile` of type `numpy.ndarray` was passed, so '
                            '`metadata` must not be `None`.')

        fname_hdr = self._get_fname_hdr(
                fname_hdr, ext=ext, interleave=interleave)

        metadata['interleave'] = interleave
        metadata['label'] = os.path.basename(
                os.path.splitext(fname_hdr)[0])
        metadata = self.tools.clean_md_sets(metadata=metadata)
        try:
            envi.save_image(fname_hdr, spyfile, dtype=dtype, force=force,
                            ext=ext, interleave=interleave,
                            byteorder=byteorder, metadata=metadata)
        except envi.EnviException:
            msg = ('Header file already exists: {0}\nUse the `force` keyword '
                   'to force overwrite.\n``hsio.set_io_defaults(force=True)`` '
                   'will adjust the default setting so existing files are '
                   'overwritten by default without passing the `force` '
                   'keyword'.format(fname_hdr))
            raise envi.EnviException(msg)

    def write_spec(self, fname_hdr_spec, df_mean, df_std, metadata=None,
                   dtype=None, force=None, ext=None, interleave=None,
                   byteorder=None):
        '''
        Wrapper function that accesses the `Spectral Python`_ package to save a
        single spectra to file.

        Parameters:
            fname_hdr_spec (``str``): Output header file path (with the '.hdr'
                extension). If the extension is explicitely specified in
                ``fname_hdr_spec`` and the ``ext`` parameter is also specified,
                ``fname_hdr_spec`` will be modified to conform to the extension
                set using the ``ext`` parameter.
            df_mean (``pandas.Series`` or ``numpy.ndarray``): Mean spectra,
                stored as a df row, where columns are the bands.
            df_std (``pandas.Series`` or ``numpy.ndarray``): Standard deviation
                of each spectra, stored as a df row, where columns are the
                bands. This will be saved to the .hdr file.
            dtype (``numpy.dtype`` or ``str``): The data type with which to
                store the image. For example, to store the image in 16-bit
                unsigned integer format, the argument could be any of
                numpy.uint16, 'u2', 'uint16', or 'H' (default=np.float32).
            force (``bool``): If ``hdr_file`` or its associated image file
                exist, ``force=True`` will overwrite the files; otherwise, an
                exception will be raised if either file exists (default=False).
            ext (``None`` or ``str``): The extension to use for saving the
                image file. If not specified or if set to an empty string
                (e.g., ``ext=''``), a default extension is determined using the
                same name as ``fname_hdr_spec``, except without the ".hdr"
                extension. If ``fname_hdr_spec`` is provided without the
                "non-.hdr" extension (e.g., "bip"), then the extension is
                determined from the ``interleave`` parameter. For example, if
                ``interleave``='bip', then ``ext`` is set to 'bip' as well. Use
                of ``ext`` is not recommended; instead, just set
                ``fname_hdr_spec`` with the correct extension or use
                ``interleave`` to set the extension (default: ``None``;
                determined from ``fname_hdr_spec`` or ``interleave``).
            interleave (``str``): The band interleave format to use for writing
                the file; ``interleave`` should be one of 'bil', 'bip', or
                'bsq' (default='bip').
            byteorder (``int`` or ``str``): Specifies the byte order
                (endian-ness) of the data as written to disk. For little
                endian, this value should be either 0 or 'little'. For big
                endian, it should be either 1 or 'big'. If not specified,
                native byte order will be used (default=None).
            metadata (``dict``): Metadata to write to the ENVI .hdr file
                describing the spectra being saved; if ``None``, will try to
                pull metadata template from ``hsio.spyfile_spec.metadata`` or
                ``hsio.spyfile.metadata`` (default=None).

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio # load hsio
            >>> fname_hdr_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio()  # initialize the hsio class (note there are no required parameters)
            >>> io.read_cube(fname_hdr_in)

            Calculate spectral mean via ``hstools.mean_datacube``

            >>> spec_mean, spec_std, _ = io.tools.mean_datacube(io.spyfile)
            >>> fname_hdr_spec = r'F:\\nigo0024\Documents\hs_process_demo\hsio\Wells_rep2_20180628_16h56m_pika_gige_7-mean.spec.hdr'

            Save the new spectra to file via ``hsio.write_spec``

            >>> io.write_spec(fname_hdr_spec, spec_mean, spec_std)
            Saving F:\nigo0024\Documents\hs_process_demo\hsio\Wells_rep2_20180628_16h56m_pika_gige_7-mean.spec

            Open *Wells_rep2_20180628_16h56m_pika_gige_7-mean.spec* in *Spectronon* for visualization

            .. image:: ../img/utilities/write_spec.png

        .. _Spectral Python: http://www.spectralpython.net/
        '''
        if dtype is None:
            dtype = self.defaults.envi_write['dtype']
        if force is None:
            force = self.defaults.envi_write['force']
        if ext is None:
            ext = self.defaults.envi_write['ext']
        if interleave is None:
            interleave = self.defaults.envi_write['interleave']
        if byteorder is None:
            byteorder = self.defaults.envi_write['byteorder']

        if metadata is None:
            try:
                metadata = self.spyfile_spec.metadata
            except AttributeError:
                metadata = self.spyfile.metadata

        fname_hdr_spec = self._get_fname_hdr(
                fname_hdr_spec, ext=ext, interleave=interleave)

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
        metadata['label'] = os.path.basename(
                os.path.splitext(fname_hdr_spec)[0])
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
        try:
            envi.save_image(fname_hdr_spec, array, interleave=interleave,
                            dtype=dtype, byteorder=byteorder,
                            metadata=metadata, force=force, ext=ext)
        except envi.EnviException:
            msg = ('Header file already exists: {0}\nUse the `force` keyword '
                   'to force overwrite.\n``hsio.set_io_defaults(force=True)`` '
                   'will adjust the default setting so existing files are '
                   'overwritten by default (i.e., without passing the `force` '
                   'keyword.)'.format(fname_hdr_spec))
            raise envi.EnviException(msg)

    def write_tif(self, fname_tif, spyfile=None, metadata=None, fname_in=None,
                  projection_out=None, geotransform_out=None,
                  show_img=False):
        '''
        Wrapper function that accesses the `GDAL Python package`_ to save a
        small datacube subset (i.e., three bands or less) to file.

        Parameters:
            fname_tif (``str``): Output image file path (with the '.tif'
                extension).
            spyfile (``SpyFile`` object or ``numpy.ndarray``, optional): The
                data cube to save. If ``numpy.ndarray``, then metadata
                (``dict``) should also be passed. If ``None``, uses
                hsio.spyfile (default: ``None``).
            metadata (``dict``): Metadata information; if ``geotransform_out``
                is not passed, "map info" is accessed from ``metadata`` and
                ``geotransform_out`` is created from that "map info".
            fname_in (``str``, optional): The filename of the image datacube to
                be read in initially. This is potentially useful if
                ``projection_out`` and/or ``geotransform_out`` are not passed
                and a ``numpy.ndarray`` is passed as the ``spyfile`` - in this
                case, ``write_tif()`` uses ``fname_in`` to load the
                ``fname_in`` datacube via GDAL, which can in turn be used to
                load the projection or geotransform information for the output
                geotiff (default: None).
            projection_out (``str``): The GDAL projection to use while writing
                the geotiff. Applied using
                gdal.driver.dataset.SetProjection() (default: ``None``;
                ``hsio.projection_out``)
            geotransform_out (``str``): The GDAL geotransform to use while
                writing the geotiff. Applied using
                gdal.driver.dataset.SetGeoTransform() (default: ``None``;
                ``hsio.geotransform_out``)
            show_img (``bool`` or ``str``): Whether to display a render of the
                image being saved as a geotiff. Must be ``False`` (does not
                display the image), "inline" (displays the image inline using
                the IPython console), or "popout" (displays the image in a
                pop-out window; default: "inline").

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio  # load hsio
            >>> fname_hdr_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio()  # initialize the hsio class
            >>> io.read_cube(fname_hdr_in)

            Save an RGB render of the datacube to file via ``hsio.write_tif``

            >>> fname_tif = r'F:\\nigo0024\Documents\hs_process_demo\hsio\Wells_rep2_20180628_16h56m_pika_gige_7.tif'
            >>> io.write_tif(fname_tif, spyfile=io.spyfile, fname_in=fname_hdr_in)
            Either `projection_out` is `None` or `geotransform_out` is `None` (or both are). Retrieving projection and geotransform information by loading `hsio.fname_in` via GDAL. Be sure this is appropriate for the data you are trying to write.
            Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

            .. image:: ../img/utilities/write_tif.png

            Open *Wells_rep2_20180628_16h56m_pika_gige_7.tif* in *QGIS* with the plot boundaries overlaid

            .. image:: ../img/utilities/write_tif_qgis.png

        .. _GDAL Python package: https://pypi.org/project/GDAL/
        '''
        if self._check_data_size(spyfile, func='write_tif',
                                 fname_out=fname_tif) is False:
            return

        msg1 = ('The directory passed in `fname_tif` does not exist. Please '
                'be sure to create the directory prior to writing the geotif.'
                '\nTry:\n'
                'os.mkdir(os.path.dirname(fname_tif))'
                ''.format(os.path.dirname(fname_tif)))
        assert os.path.isdir(os.path.dirname(fname_tif)) is True, msg1
        msg2 = ('`show_img` must be one of `False`, "inline", or "popout". '
                'Please modify `show_img` paramater accordingly.\n')
        assert show_img in [False, 'inline', 'popout', None], msg2

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
                  'geotransform information by loading `hsio.fname_in` via '
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
            msg3 = ('GDAL driver was unable to successfully create the empty '
                   'geotiff; check to be sure the correct filename is being '
                   'passed: {0}\n'.format(fname_tif))
            assert tif_out is not None, msg3
            tif_out.SetProjection(projection_out)
            tif_out.SetGeoTransform(geotransform_out)

            band_b = self.tools.get_band(460)
            band_g = self.tools.get_band(550)
            band_r = self.tools.get_band(640)
            band_list = [band_r, band_g, band_b]  # backwards for RGB display
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
                band_out = None
            if show_img == 'inline':
                self.show_img(array, band_r=band_r, band_g=band_g,
                              band_b=band_b, cbar=False, inline=True)
            elif show_img == 'popout':
                self.show_img(array, band_r=band_r, band_g=band_g,
                              band_b=band_b, cbar=False, inline=False)
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
            if show_img == 'inline':
                self.show_img(array, cbar=True, inline=True)
            elif show_img == 'popout':
                self.show_img(array, cbar=True, inline=False)

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
    def __init__(self, spyfile):
        msg = ('Pleae load a SpyFile (Spectral Python object)')
        assert spyfile is not None, msg

        self.spyfile = spyfile

        self.fname_in = None
        self.fname_hdr = None
        self.base_name = None
        self.name_short = None
        self.name_long = None
        self.name_plot = None
        self.meta_bands = None

        self.load_spyfile(spyfile)

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

    def _parse_fname_plot(self, str_plot):
        '''
        Code for parsing ``name_plot`` (numeric text following ``str_plot``).
        '''
        s = self.name_short
        if str_plot in s and '_pika' in s:
            name_plot = s[s.find(str_plot) + len(str_plot):s.find('_pika')]
        elif str_plot in s and '_pika' not in s:
            name_plot = s[s.find(str_plot) + len(str_plot):s.find('-')]
        else:
            name_plot = self.name_short.rsplit('_', 1)[1]

        if len(name_plot) > 12:  # then it must have gone wrong
            name_plot = self.name_short.rsplit('_', 1)[1]
        if len(name_plot) == 0:  # then '_pika' must not
            name_plot = self.name_short.rsplit('_', 1)[1]

        try:
            int(name_plot)
        except ValueError:  # give up..
            msg = ('Cannot determine the plot name from the image filename. '
                   'Setting `hsio.name_plot` to `None`. If this image is for '
                   'a particular plot, please set `hsio.name_plot; otherwise, '
                   'ignore this warning.\n')
            warnings.warn(msg, UserWarning)

            name_plot = None
        return name_plot

    def _parse_fname(self, fname_hdr=None, str_plot='plot_', overwrite=True,
                     name_long=None, name_short=None, name_plot=None):
        '''
        Parses the filename for ``name_long`` (text after the first dash,
        inclusive), ``name_short`` (text before the first dash), and
        ``name_plot`` (numeric text following ``str_plot``).

        Parameters:
            fname_hdr (``str``): input filename.
            str_plot (``str``): text to search for that precedes the numeric
                text that describes the plot number.
            overwrite (``bool``): whether the class instances of ``name_long``,
                ``name_short``, and ``name_plot`` should be overwritten based
                on ``fname_in`` (default: ``True``).
        '''
        if fname_hdr is None:
            fname_hdr = self.spyfile.filename + '.hdr'
        if os.path.splitext(fname_hdr)[1] == '.hdr':  # modify self.fname_in based on new file
            fname_in = os.path.splitext(fname_hdr)[0]
        else:
            fname_hdr = fname_hdr + '.hdr'
            fname_in = os.path.splitext(fname_hdr)[0]
        self.fname_in = fname_in
        self.fname_hdr = fname_hdr

        self.base_dir = os.path.dirname(fname_in)
#        base_name = os.path.basename(fname_in)
        base_name = os.path.basename(os.path.splitext(fname_in)[0])
        self.base_name = base_name

        if overwrite is True:
            if '-' in base_name:
                self.name_long = base_name[base_name.find(
                        '-', base_name.rfind('_')):]
                self.name_short = base_name[:base_name.find(
                        '-', base_name.rfind('_'))]
            else:
                # if name_long does not have ext, it can be just blank
                self.name_long = ''
                # and name_short can be base_name
                self.name_short = base_name
            self.name_plot = self._parse_fname_plot(str_plot)

        if name_long is not None:
            self.name_long = name_long
        elif overwrite is False and self.name_long is None:
            if '-' in base_name:
                self.name_long = base_name[base_name.find(
                        '-', base_name.rfind('_')):]
            else:
                # if name_long does not have ext, it can be just blank
                self.name_long = ''
        else:
            pass

        if name_short is not None:
            self.name_short = name_short
        elif overwrite is False and self.name_short is None:
            if '-' in base_name:
                self.name_short = base_name[:base_name.find(
                        '-', base_name.rfind('_'))]
            else:
                self.name_short = base_name
        else:
            pass

        if name_plot is not None:
            self.name_plot = name_plot
        elif overwrite is False and self.name_plot is None:
            self.name_plot = self._parse_fname_plot(str_plot)
        else:
            pass

    def clean_md_sets(self, metadata=None):
        '''
        Converts metadata items that are expressed as a list to be expressed as
        a dictionary.

        Parameters:
            metadata (``dict``, optional): Metadata dictionary to clean

        Returns:
            ``dict``:
                **metadata_out** (``dict``) -- Cleaned metadata dictionary.

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Create sample metadata with "wavelength" expressed as a list of strings

            >>> metadata = {'samples': 1300,
                            'lines': 617,
                            'bands': 4,
                            'file type': 'ENVI Standard',
                            'wavelength': ['394.6', '396.6528', '398.7056',
                            '400.7584']}

            Clean metadata using ``hstools.clean_md_sets``. Notice how
            wavelength is now expressed as a ``str`` representation of a
            ``dict``, which is required for properly writing the metadata to
            the .hdr file via `save_image()`_ in Spectral Python.

            >>> io.tools.clean_md_sets(metadata=metadata)
            {'samples': 1300,
             'lines': 617,
             'bands': 4,
             'file type': 'ENVI Standard',
             'wavelength': '{394.6, 396.6528, 398.7056, 400.7584}'}

        .. _save_image(): http://www.spectralpython.net/class_func_ref.html#spectral.io.envi.save_image
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

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Create sample metadata

            >>> metadata = {'samples': 1300,
                            'lines': 617,
                            'bands': 4,
                            'file type': 'ENVI Standard',
                            'map info': '{UTM, 1.0, 1.0, 421356.76707299997, 4844936.7317699995, 0.04, 0.04, 15, T, WGS-84, units  meters, rotation  0.000}',
                            'wavelength': ['394.6', '396.6528', '398.7056',
                            '400.7584']}

            Delete *"map info"* from ``metadata`` using ``hstools.del_met_item``

            >>> io.tools.del_meta_item(metadata, 'map info')
            {'samples': 1827,
             'lines': 617,
             'bands': 4,
             'file type': 'ENVI Standard',
             'wavelength': ['394.6', '396.6528', '398.7056', '400.7584']}
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

    def dir_data(self,):
        '''Retrieves the data directory from "site packages".'''
        dir_data = os.path.join(sysconfig.get_paths()['purelib'], 'hs_process', 'data')
        return dir_data

    def get_band(self, target_wl, spyfile=None):
        '''
        Finds the band number of the closest target wavelength.

        Parameters:
            target_wl (``int`` or ``float``): the target wavelength to retrieve
                the band number for (required).
            spyfile (``SpyFile`` object, optional): The datacube being accessed
                and/or manipulated; if ``None``, uses ``hstools.spyfile``
                (default: ``None``).

        Returns:
            ``int``:
                **key_band** (``int``) -- band number of the closest target
                wavelength (``target_wl``).

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Use ``hstools.get_band`` to find the band number corresponding to
            *703 nm*

            >>> io.tools.get_band(703, io.spyfile)
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
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Use ``hstools.get_wavelength`` to find the wavelength value corresponding to the *151st band*

            >>> io.tools.get_wavelength(151, io.spyfile)
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

    def get_wavelength_range(self, range_bands, index=True, spyfile=None):
        '''
        Retrieves the wavelengths for all bands within a band range.

        Parameters:
            range_bands (``list``): the minimum and maximum band number to
                consider; values should be ``int``.
            index (bool): Indicates whether the bands in ``range_bands`` denote
                the band number (``False``; min=1) or the index number
                (``True``; min=0) (default: ``True``).

        Returns:
            ``list``:
                **wavelength_list** (``list``): A list of all wavelengths
                between a range in band numbers or index values (depending how
                ``index`` is set).

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_hdr = r'F:\nigo0024\Documents\GitHub\hs_process\hs_process\data\Wells_rep2_20180628_16h56m_test_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_hdr)

            Find the wavelengths from the *16th* to *21st bands*

            >>> io.tools.get_wavelength_range([16, 21], index=False, spyfile=io.spyfile)
            [425.392, 427.4448, 429.4976, 431.5504, 433.6032, 435.656]

            Find the wavelengths from the *16th* to the *21st index*

            >>> io.tools.get_wavelength_range([16, 21], index=True, spyfile=io.spyfile)
            [427.4448, 429.4976, 431.5504, 433.6032, 435.656, 437.7088]
        '''
        msg = ('"range_bands" must be a `list` or `tuple`.')
        assert isinstance(range_bands, list) or isinstance(range_bands, tuple), msg
        # could also just take the min and max as long as it has at least 2..
        msg = ('"range_bands" must have exactly two items.')
        assert len(range_bands) == 2, msg

        range_bands = sorted(range_bands)
        if index is True:
            range_bands[0] += 1
            range_bands[1] += 1

        wl_min = self.get_wavelength(min(range_bands))  # gets closest wavelength
        wl_max = self.get_wavelength(max(range_bands))
        band_min = self.get_band(wl_min)
        band_max = self.get_band(wl_max)

        if band_min < min(range_bands):  # ensures its actually within the range
            band_min += 1
            wl_min = self.get_wavelength(band_min)
        if band_max > max(range_bands):
            band_max -= 1
            wl_max = self.get_wavelength(band_max)
        wl_list = [self.get_wavelength(x) for x in range(band_min, band_max+1)]
        return wl_list

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

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Using ``hstools.get_center_wl``, find the bands and *actual mean wavelength* of the bands closest to *700* and *710* nm.

            >>> bands, wls_mean = io.tools.get_center_wl([700, 710], wls=True)
            >>> bands
            [150, 155]
            >>> wls_mean
            705.5992
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
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Using ``hstools.get_band_index``, find the band index of the *4th*, *43rd*, and *111th* bands

            >>> io.tools.get_band_index([4, 43, 111])
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
        Gets the spectral mean of a datacube from a list of bands.

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

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Calculate the spectral mean of the datacube via
            ``hstools.get_spectral_mean`` using all bands between *800* and
            *840 nm*

            >>> band_list = io.tools.get_band_range([800, 840], index=False)
            >>> array_mean = io.tools.get_spectral_mean(band_list, spyfile=io.spyfile)
            >>> io.show_img(array_mean)

            .. image:: ../img/utilities/get_spectral_mean.png

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
            band_idx (``int`` or ``list``): the target band index(es) to
                retrive the band number for (required).

        Returns:
            ``int`` or ``list``:
                **band_num** (``int`` or ``list``): band number of the passed
                band index (``band_idx``).

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Using ``hstools.get_band_num``, find the band number located at the *4th*, *43rd*, and *111th* index values.

            >>> io.tools.get_band_num([4, 43, 111])
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
            index (bool): Indicates whether to return the band number
                (``False``; min=1) or to return index number (``True``; min=0)
                (default: ``True``).

        Returns:
            ``list``:
                **band_list** (``list``): A list of all bands (either index or
                number, depending on how ``index`` is set) between a range in
                wavelength values.

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Find the band number of all bands between *700* and *710 nm*

            >>> io.tools.get_band_range([700, 710], index=False, spyfile=io.spyfile)
            [150, 151, 152, 153, 154]

            Find the band index values of all bands between *700* and *710 nm*
            via ``hstools.get_band_range``

            >>> io.tools.get_band_range([700, 710], index=True, spyfile=io.spyfile)
            [149, 150, 151, 152, 153]
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
                **metadata_list** (``list`` or ``str``): List of metadata set
                items (as ``str``), or if idx is not ``None``, the item in the
                position described by ``idx``.

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Retrieve the *"map info" set* from the metadata via
            ``hstools.get_meta_set``

            >>> map_info_set = io.spyfile.metadata['map info']
            ['UTM',
             '1.0',
             '1.0',
             '441357.287073',
             '4855944.7717699995',
             '0.04',
             '0.04',
             '15',
             'T',
             'WGS-84',
             'units  meters',
             'rotation  0.000']
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
            pix_e_ul (``int``): upper left column (easting) where image
                cropping begins.
            pix_n_ul (``int``): upper left row (northing) where image cropping
                begins.
            utm_x (``float``): UTM easting coordinates (meters) of the original
                image (from the upper left).
            utm_y (``float``): UTM northing coordinates (meters) of the
                original image (from the upper left).
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

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Retrieve UTM coordinates and pixel sizes from the metadata

            >>> map_info_set = io.spyfile.metadata['map info']
            >>> utm_x = io.tools.get_meta_set(map_info_set, 3)
            >>> utm_y = io.tools.get_meta_set(map_info_set, 4)
            >>> spy_ps_e = float(map_info_set[5])
            >>> spy_ps_n = float(map_info_set[6])

            Calculate the UTM coordinates at the *100th easting pixel* and
            *50th northing pixel* using ``hstools.get_UTM``

            >>> ul_x_utm, ul_y_utm = io.tools.get_UTM(100, 50,
                                                      utm_x, utm_y,
                                                      spy_ps_e,
                                                      spy_ps_n)
            >>> ul_x_utm
            441360.80707299995
            >>> ul_y_utm
            4855934.691769999
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

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Load a new datacube using ``hstools.load_spyfile``

            >>> io.tools.load_spyfile(io.spyfile)
            >>> io.tools.spyfile
            Data Source:   'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip'
        	# Rows:            617
        	# Samples:        1300
        	# Bands:           240
        	Interleave:        BIP
        	Quantization:  32 bits
        	Data format:   float32
        '''
        self.spyfile = spyfile
        self._get_meta_bands(spyfile)
        self._parse_fname(fname_hdr=None, str_plot='plot_', overwrite=True,
                          name_long=None, name_short=None, name_plot=None)

    def mask_array(self, array, metadata, thresh=None, percentile=None,
                   side='lower'):
        '''
        Creates a masked numpy array based on a threshold value. If ``array`` is
        already a masked array, that mask is maintained and the new mask(s) is/
        are added to the original mask.

        Parameters:
            array (``numpy.ndarray``): The data array to mask.
            thresh (``float`` or ``list``): The value for which to base the
                threshold; if ``thresh`` is ``list`` and ``side`` is ``None``,
                then all values in ``thresh`` will be masked; if ``thresh`` is
                ``list`` and ``side`` is not ``None``, then only the first
                value in the list will be considered for thresholding (default:
                ``None``).
            percentile (``float``): The percentile of pixels to mask; if
                ``percentile`` = 95 and ``side`` = 'lower', the lowest 95% of
                pixels will be masked prior to calculating the mean spectra
                across pixels (default: ``None``; range: 0-100).
            side (``str``): The side of the threshold for which to apply the
                mask. Must be either 'lower', 'upper', 'outside', or ``None``;
                if 'lower', everything below the threshold will be masked; if
                'outside', the ``thresh`` / ``percentile`` parameter must be
                list-like with two values indicating the lower and upper bounds
                - anything outside of these values will be masked out; if
                ``None``, only the values that exactly match the threshold will
                be masked (default: 'lower').

        Returns:
            2-element ``tuple`` containing

            - **array_mask** (``numpy.ndarray``): The masked ``numpy.ndarray``
              based on the passed threshold and/or percentile value.
            - **metadata** (``dict``): The modified metadata.

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Retrieve the image band at *800 nm* using ``hstools.get_band`` and
            ``hsio.spyfile.open_memmap``

            >>> band = io.tools.get_band(800)
            >>> array = io.spyfile.open_memmap()[:, :, band]

            Create a masked array of all values below the *75th percentile*
            via ``hstools.mask_array``

            >>> array_mask, metadata = io.tools.mask_array(array, io.spyfile.metadata, percentile=75, side='lower')

            See that the *"history"* tage in the ``metadata`` has been modified

            >>> metadata['history'][-158:]
            "hs_process.mask_array[<label: 'thresh?' value:None; label: 'percentile?' value:75; label: 'side?' value:lower; label: 'unmasked_pct?' value:24.9935170178282>]"

            Visualize the unmasked array using ``hsio.show_img``. Set ``vmin``
            and ``vmax`` to ensure the same color scale is used in comparing
            the masked vs. unmasked arrays.

            >>> vmin = array.min()
            >>> vmax = array.max()
            >>> io.show_img(array, vmin=vmin, vmax=vmax)

            .. image:: ../img/utilities/mask_array_800nm.png

            Visualize the unmasked array using ``hsio.show_img``

            >>> io.show_img(array_mask, vmin=vmin, vmax=vmax)

            .. image:: ../img/utilities/mask_array_800nm_75th.png
        '''
        msg1 = ('``side`` must be one of the following: "lower", "upper", '
                '"outside", or "equal".')
        msg2 = ('``side`` is {0}, so either ``percentile`` or ``thresh`` must '
                'be list-type with length of two (the lower and upper bounds)')
        assert side in ['lower', 'upper', 'outside', 'equal'], msg1
        if side == 'outside':
            assert isinstance(percentile, list), msg2
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
            elif side == 'outside':
                if len(array_pctl) > 2:
                    print('WARNING: There were more than two percentile '
                          'values passed to ``hstools.mask_array``. Using '
                          'only the first two values (after sorting).')
                msg = ('Two percentile values must be passed to '
                       '``hstools.mask_array``. ``percentile``: {0}'
                       ''.format(percentile))
                assert isinstance(array_pctl, np.ndarray), msg
                array_pctl.sort()
                mask_array_p = np.ma.masked_less_equal(array, array_pctl[0])
                mask_array_p = np.ma.masked_greater(mask_array_p, array_pctl[1])
            elif side is None:
                mask_array_p = np.ma.masked_equal(array, array_pctl)
        else:
            mask_array_p = np.ma.masked_less(array, np.nanmin(array)-1e-6)
        if thresh is not None:
            if side == 'lower':
                mask_array_t = np.ma.masked_less_equal(array, thresh)
            elif side == 'upper':
                mask_array_t = np.ma.masked_greater(array, thresh)
            elif side == 'outside':
                msg = ('Two threshold values must be passed to '
                       '``hstools.mask_array`` in a list-like object. '
                       '``thresh``: {0}'.format(thresh))
                assert isinstance(thresh, list) or isinstance(thresh, tuple), msg
                if len(thresh) > 2:
                    print('WARNING: There were more than two threshold '
                          'values passed to ``hstools.mask_array``. Using '
                          'only the first two values (after sorting).')
                thresh.sort()
                mask_array_t = np.ma.masked_less_equal(array, thresh[0])
                mask_array_t = np.ma.masked_greater(mask_array_t, thresh[1])
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
        array_mask = np.ma.array(array, mask=mask_combine)  # combines aray (and its mask) with mask_combine (masks all cells with a mask in either)
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
        try:
            metadata['history'] += hist_str
        except KeyError:
            metadata['history'] = hist_str[4:]
        return array_mask, metadata

#    def mask_datacube(self, spyfile, mask):
#        '''
#        DO NOT USE; USE mean_datacube() INSTEAD AND PASS A MASK.
#
#
#        Applies ``mask`` to ``spyfile``, then returns the datcube (as a np.array)
#        and the mean spectra
#
#        Parameters:
#            spyfile (``SpyFile`` object or ``numpy.ndarray``): The hyperspectral
#                datacube to mask.
#            mask (``numpy.ndarray``): the mask to apply to ``spyfile``; if ``mask``
#                does not have similar dimensions to ``spyfile``, the first band
#                (i.e., first two dimensions) of ``mask`` will be repeated n times
#                to match the number of bands of ``spyfile``.
#        '''
#        if isinstance(spyfile, SpyFile.SpyFile):
#            self.load_spyfile(spyfile)
#            array = self.spyfile.load()
#        elif isinstance(spyfile, np.ndarray):
#            array = spyfile.copy()
#
#        if isinstance(mask, np.ma.masked_array):
#            mask = mask.mask
#        if mask.shape != spyfile.shape:
#            if len(mask.shape) == 3:
#                mask_2d = np.reshape(mask, mask.shape[:2])
#            else:
#                mask_2d = mask.copy()
#            mask = np.empty(spyfile.shape)
#            for band in range(spyfile.nbands):
#                mask[:, :, band] = mask_2d
#
#        datacube_masked = np.ma.masked_array(array, mask=mask)
#        spec_mean = np.nanmean(datacube_masked, axis=(0, 1))
#        spec_std = np.nanstd(datacube_masked, axis=(0, 1))
#        spec_mean = pd.Series(spec_mean)
#        spec_std = pd.Series(spec_std)
#        return spec_mean, spec_std, datacube_masked

    def mean_datacube(self, spyfile, mask=None, nodata=0):
        '''
        Calculates the mean spectra for a datcube; if ``mask`` is passed (as a
        ``numpy.ndarray``), then the mask is applied to ``spyfile`` prior to
        computing the mean spectra.

        Parameters:
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The
                hyperspectral datacube to mask.
            mask (``numpy.ndarray``): the mask to apply to ``spyfile``; if
                ``mask`` does not have similar dimensions to ``spyfile``, the
                first band (i.e., first two dimensions) of ``mask`` will be
                repeated *n* times to match the number of bands of ``spyfile``
                (default: ``None``).
            nodata (``float`` or ``None``): If ``None``, treats all pixels
                cells as they are repressented in the ``numpy.ndarray``.
                Otherwise, replaces ``nodata`` with ``np.nan`` and these cells
                will not be considered when calculating the mean spectra.

        Returns:
            3-element ``tuple`` containing

            - **spec_mean** (``SpyFile.SpyFile`` object): The mean spectra from
              the input datacube.
            - **spec_std** (``SpyFile.SpyFile`` object): The standard deviation
              of the spectra from the input datacube.
            - **datacube_masked** (``numpy.ndarray``): The masked
              ``numpy.ndarray``; if ``mask`` is ``None``, ``datacube_masked``
              is identical to the ``SpyFile`` data array.

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Retrieve the image band at *800 nm* using ``hstools.get_band`` and
            ``hsio.spyfile.open_memmap``, then mask out all pixels whose value falls
            below the *75th percentile*.

            >>> band = io.tools.get_band(800)
            >>> array = io.spyfile.open_memmap()[:, :, band]
            >>> array_mask, metadata = io.tools.mask_array(array, io.spyfile.metadata, percentile=75, side='lower')

            Calculate the spectral mean from the remaining *(i.e., unmasked)*
            pixels using ``hstools.mean_datacube``.

            >>> spec_mean, spec_std, datacube_masked = io.tools.mean_datacube(io.spyfile, mask=array_mask)

            Save using ``hsio.write_spec`` and ``hsio.write_cube``, then load
            into Spectronon software for visualization.

            >>> fname_hdr_spec = r'F:\\nigo0024\Documents\hs_process_demo\hstools\Wells_rep2_20180628_16h56m_pika_gige_7-mean_800nm_75th.spec.hdr'
            >>> fname_hdr_cube = r'F:\\nigo0024\Documents\hs_process_demo\hstools\Wells_rep2_20180628_16h56m_pika_gige_7-mean_800nm_75th.bip.hdr'
            >>> io.write_spec(fname_hdr_spec, spec_mean, spec_std, metadata=metadata)
            Saving F:\nigo0024\Documents\hs_process_demo\hstools\Wells_rep2_20180628_16h56m_pika_gige_7-mean_800nm_75th.spec
            >>> io.write_cube(fname_hdr_cube, datacube_masked, metadata=metadata)
            Saving F:\nigo0024\Documents\hs_process_demo\hstools\Wells_rep2_20180628_16h56m_pika_gige_7-mean_800nm_75th.bip

            .. image:: ../img/utilities/mean_datacube.png
        '''
        if isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = self.spyfile.load()
            # array = self.spyfile.open_memmap().copy()
            nbands = spyfile.nbands
            shape = spyfile.shape
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()
            if len(array.shape) == 3:
                nbands = array.shape[2]
            else:
                nbands = 1
            shape = array.shape
        array[array==nodata] = np.nan

        if mask is None:  # find all invalid values and mask them
            if nbands == 1:
                mask = np.ma.masked_greater(array[:, :], 1e10).mask
            else:
                mask = np.ma.masked_greater(array[:, :, 0], 1e10).mask
        if isinstance(mask, np.ma.masked_array):
            mask = mask.mask

        if mask is not None:
            if mask.shape != shape:
                if len(mask.shape) == 3:
                    mask_2d = np.reshape(mask, mask.shape[:2])
                else:
                    mask_2d = mask.copy()
                mask = np.empty(shape)
                for band in range(nbands):
                    mask[:, :, band] = mask_2d

        datacube_masked = np.ma.masked_array(array, mask=mask)
        # spec_mean = np.mean(datacube_masked, axis=(0, 1))
        # spec_std = np.std(datacube_masked, axis=(0, 1))
        spec_mean = np.nanmean(datacube_masked, axis=(0, 1))
        spec_std = np.nanstd(datacube_masked, axis=(0, 1))
        spec_mean = pd.Series(spec_mean)
        spec_std = pd.Series(spec_std)
        return spec_mean, spec_std, datacube_masked

        # adjust any values that are nan or inf
        # spec_mean[0]
        # try:
        #     i = np.where(spec_mean == -np.inf)[0]
        #     i = np.where(spec_mean == None)[0]
        # except ValueError:
        #     pass

        # i = np.where(spec_mean > 100)[0]


#    def mask_shadow(self, shadow_pctl=20, show_histogram=False,
#                    spyfile=None):
#        '''
#        Creates a ``numpy.mask`` of all pixels that are likely shadow pixels.
#
#        Parameters:
#            shadow_pctl (``int``): the percentile of pixels in the image to mask
#                (default: 20).
#            show_histogram (``bool``):
#            spyfile (``SpyFile.SpyFile`` object):
#
#        Returns:
#            2-element ``tuple`` containing
#
#            - **array_noshadow.mask** (``numpy.mask``): The mask indicating all
#              pixels that are likely shadow pixels.
#            - **metadata** (``dict``): The modified metadata.
#
#        Example:
#            >>> from hs_process import hsio
#            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
#            >>> io = hsio(fname_in)
#            >>> io.tools.mask_shadow(shadow_pctl=20, show_histogram=False,
#                    spyfile=None)
#        '''
#        if spyfile is None:
#            spyfile = self.spyfile
#            array = self.spyfile.load()
#        elif isinstance(spyfile, SpyFile.SpyFile):
#            self.load_spyfile(spyfile)
#            array = self.spyfile.load()
#        elif isinstance(spyfile, np.ndarray):
#            array = spyfile.copy()
#
#        array_energy = np.mean(array, axis=2)
#        array_noshadow, metadata = self.mask_array(
#                array_energy, self.spyfile.metadata, percentile=shadow_pctl,
#                side='lower', show_histogram=show_histogram)
#        return array_noshadow.mask, metadata

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

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Retrieve the *"map info" set* from the metadata via
            ``hstools.get_meta_set``

            >>> map_info_set = io.spyfile.metadata['map info']
            >>> map_info_set
            ['UTM',
             '1.0',
             '1.0',
             '441357.287073',
             '4855944.7717699995',
             '0.04',
             '0.04',
             '15',
             'T',
             'WGS-84',
             'units  meters',
             'rotation  0.000']

            Modify the value at *index position 4* from ``4855944.7717699995``
            to ``441300.2`` using ``hstools.modify_meta_set``.

            >>> io.tools.modify_meta_set(map_info_set, idx=4, value=441300.2)
            '{UTM, 1.0, 1.0, 441357.287073, 441300.2, 0.04, 0.04, 15, T, WGS-84, units  meters, rotation  0.000}'
        '''
        metadata_list = self.get_meta_set(meta_set, idx=None)
        metadata_list[idx] = str(value)
        set_str = '{' + ', '.join(str(x) for x in metadata_list) + '}'
        return set_str

    def plot_histogram(self, array, fname_fig=None, title=None, xlabel=None,
                        percentile=90, bins=50, fontsize=16, color='#444444'):
        '''
        Plots a histogram with the percentile value labeled.

        Parameters:
            array (``numpy.ndarray``): The data array used to create the
                histogram for; if ``array`` is masked, the masked pixels are
                excluded from the histogram.
            fname_fig (``str``, optional): The filename to save the figure to;
                if ``None``, the figure will not be saved (default: ``None``).
            title (``str``, optional): The plot title (default: ``None``).
            xlabel (``str``, optional): The x-axis label of the histogram
                (default: ``None``).
            percentile (``scalar``, optional): The percentile to label and
                illustrate on the histogram; if ``percentile`` = 90, the
                band/spectral index value at the 90th percentile will be
                labeled on the plot (default: 90; range: 0-100).
            bins (``int``, optional): Number of histogram bins (default: 50).
            fontsize (``scalar``): Font size of the axes labels. The title and
                text annotations will be scaled relatively (default: 16).
            color (``str``, optional): Color of the histogram columns (default:
                "#444444")

        Returns:
            fig (``matplotlib.figure``): Figure object showing the histogram.

        Example:
            Load and initialize ``hsio``

            >>> from hs_process import hsio
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)

            Retrieve the image band at *800 nm* using ``hstools.get_band`` and
            ``hsio.spyfile.open_memmap``

            >>> band = io.tools.get_band(800)
            >>> array = io.spyfile.open_memmap()[:, :, band]

            Create a masked array of all values below the *5th percentile*
            via ``hstools.mask_array``

            >>> array_mask, metadata = io.tools.mask_array(array, io.spyfile.metadata, percentile=5, side='lower')

            Visualize the histogram of the unmasked pixels (i.e., those greater
            than the 5th percentile) using ``hstools.plot_histogram``

            >>> title = 'Reflectance at 800 nm'
            >>> xlabel = 'Reflectance (%)'
            >>> fig = io.tools.plot_histogram(array_mask, title=title, xlabel=xlabel)

            .. image:: ../img/utilities/plot_histogram_800nm.png
        '''
        plt.close('all')  # close all other plots before create a new one
        # this is useful when this function is accesed by the batch module
        plt.style.use(plt_style)

        msg = ('Array must be 1-dimensional or 2-dimensional. Please choose '
               'only a single array band to create a histogram\nArray shape: '
               '{0}'.format(array.shape))
        if len(array.shape) == 3:
            assert array.shape[2] == 1, msg
            array = np.squeeze(array)
        else:
            assert len(array.shape) <= 2, msg

        if isinstance(array, np.ma.core.MaskedArray):
            array_m = array.compressed()  # allows for accurate percentile calc
        else:
            array_m = np.ma.masked_array(array, mask=False)
            array_m = array_m.compressed()

#        if percentile is not None:
        pctl = np.nanpercentile(array_m.flatten(), percentile)

        fig, ax = plt.subplots()
        ax = sns.distplot(array_m.flatten(), bins=bins, color='grey')
        data_x, data_y = ax.lines[0].get_data()

        y_lim = ax.get_ylim()
        yi = np.interp(pctl, data_x, data_y)
        ymax = yi/y_lim[1]
        ax.axvline(pctl, ymax=ymax, linestyle='--', color=color, linewidth=0.5)
        boxstyle_str = 'round, pad=0.5, rounding_size=0.15'

        legend_str = ('Percentile ({0}): {1:.3f}'
                      ''.format(percentile, pctl))
        ax.annotate(
            legend_str,
            xy=(pctl, yi),
            xytext=(0.97, 0.94),  # loc to place text
            textcoords='axes fraction',  # placed relative to axes
            ha='right',  # alignment of text
            va='top',
            fontsize=int(fontsize * 0.9),
            color=color,
            bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                      ec=(0.5, 0.5, 0.5), alpha=0.5),
            arrowprops=dict(arrowstyle='-|>',
                            color=color,
        #                    patchB=el,
                            shrinkA=0,
                            shrinkB=0,
                            connectionstyle='arc3,rad=-0.3',
                            linestyle='--',
                            linewidth=0.7))
        ax.set_title(title, fontweight='bold', fontsize=int(fontsize * 1.1))
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel('Frequency (%)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        plt.tight_layout()
        if fname_fig is not None:
            if not os.path.isdir(os.path.dirname(fname_fig)):
                os.mkdir(os.path.dirname(fname_fig))
            fig.savefig(fname=fname_fig, dpi=300)
        return fig
