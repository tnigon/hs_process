# -*- coding: utf-8 -*-
import numpy as np
import os
import spectral.io.spyfile as SpyFile

from hs_process.utilities import defaults
from hs_process.utilities import hstools


class segment(object):
    '''
    Class for aiding in the segmentation/masking of image data to include
    pixels that are of most interest.
    '''
    def __init__(self, spyfile):
        '''
        spyfile (`SpyFile` object): The Spectral Python datacube to manipulate.
        '''
        self.spyfile = spyfile
        self.tools = hstools(spyfile)

        self.spy_ps_e = None
        self.spy_ps_n = None
        self.spy_ul_e_srs = None
        self.spy_ul_n_srs = None

        self.defaults = defaults
        self.load_spyfile(spyfile)

    def _get_band_list(self, band, list_range):
        '''
        Determines how a "band" should be consolidated, if at all.
        '''
        if isinstance(band, list) and list_range is True:
            msg = ('When using a `list_range`, please be sure each passed '
                   '"band" is a list of exactly two wavelength values.\n')
            assert len(band) == 2, msg
            band_list = self.tools.get_band_range(band, index=False)
        elif isinstance(band, list) and list_range is False:
            band_list = []
            for b_i in band:
                b, wl = self.tools.get_band(b_i)
                band_list.append(b)
        else:  # just a single band; disregards `list_range`
            b, wl = self.tools.get_band(band)
            band_list = [b]
        return band_list

    def band_math_ratio(self, b1, b2, spyfile=None, list_range=True):
        '''
        Calculates a simple ratio spectral index from two input band
        wavelengths. Wavelength bands can be input as two individual bands, two
        sets of bands (i.e., list of bands), or range of bands (i.e., list of
        two bands indicating the lower and upper range).

        Parameters:
            b1 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the first parameter (numerator) of
                the ratio index; if `list`, then consolidates all bands between
                two wavelength values by calculating the mean pixel value
                across all bands in that range.
            b2 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the second parameter (denominator)
                of the ratio index; if `list`, then consolidates all bands between
                two wavelength values by calculating the mean pixel value
                across all bands in that range.
            spyfile (`SpyFile` object or `numpy.ndarray`): The datacube to
                crop; if `numpy.ndarray` or `None`, loads band information from
                `self.spyfile` (default: `None`).
            list_range (`bool`): Whether a band passed as a list is interpreted as a
                range of bands (`True`) or for each individual band in the
                list (`False`). If `list_range` is `True`, `b1` and `b2` should
                be lists with two items, and the first item should correspond
                to the lower range, and second item should correspond to the
                upper range (default: `True`).
        '''
        if spyfile is None:
            spyfile = self.spyfile
            array = spyfile.load()
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = spyfile.load()
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()
            spyfile = self.spyfile
        metadata = self.tools.spyfile.metadata

        band1_list = self._get_band_list(b1, list_range)
        band2_list = self._get_band_list(b2, list_range)
        array_b1 = self.tools.get_spectral_mean(band1_list, array)
        array_b2 = self.tools.get_spectral_mean(band2_list, array)
        array_ratio = (array_b1/array_b2)

        metadata['bands'] = 1
        self.tools.del_meta_item(metadata, 'wavelength')
        self.tools.del_meta_item(metadata, 'band names')
        hist_str = (" -> hs_process.band_math_ratio[<"
                    "label: 'b1?' value:{0}; "
                    "label: 'b2?' value:{1}; "
                    "label: 'list_range?' value:{2}>]"
                    "".format(b1, b2, list_range))
        metadata['history'] += hist_str
        metadata['samples'] = array_ratio.shape[1]
        metadata['lines'] = array_ratio.shape[0]
        return array_ratio, metadata

#        if name is None:
#            name = 'ratio_{0:.0f}_{1:.0f}'.format(wl1, wl2)
#        fname_out_envi = os.path.join(
#            base_dir_out, (name_print + '_' + str(name) + '.' + interleave))
#        print('Calculating normalized difference index for {0}: '
#              '{1:.0f}/{2:.0f}'.format(name_print, wl1, wl2))
#        return array_index

    def band_math_ndi(self, b1=780, b2=559, spyfile=None, list_range=True):
        '''
        Calculates a normalized difference spectral index for two bands. Bands
        can be input as two individual bands, two sets of bands (i.e., list of
        bands), or range of bands (i.e., list of two bands indicating the lower
        and upper range).

        Parameters:
            b1 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                normalized difference index; if `list`, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range.
            b2 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the second parameter of the
                normalized difference index; if `list`, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range.
            b1 (`int`, `float`, or `list`): the band (or set of bands) to be
                used as the first parameter of the normalized difference index;
                if `list`, then consolidates all bands between two wavelength
                values by calculating the mean pixel value across all bands in
                that range.
            b2 (`int`, `float`, or `list`): the band (or set of bands) to be
                used as the second parameter of the normalized difference
                index; if `list`, then consolidates all bands by calculating
                the mean pixel value across all bands in the list.
            spyfile (`SpyFile` object or `numpy.ndarray`): The datacube to
                crop; if `numpy.ndarray` or `None`, loads band information from
                `self.spyfile` (default: `None`).
            list_range (`bool`): Whether a band passed as a list is interpreted as a
                range of bands (`True`) or for each individual band in the
                list (`False`). If `list_range` is `True`, `b1` and `b2` should
                be lists with two items, and the first item should correspond
                to the lower range, and second item should correspond to the
                upper range (default: `True`).
            '''
        if spyfile is None:
            spyfile = self.spyfile
            array = spyfile.load()
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = spyfile.load()
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()
            spyfile = self.spyfile
        metadata = self.tools.spyfile.metadata

        band1_list = self._get_band_list(b1, list_range)
        band2_list = self._get_band_list(b2, list_range)
        array_b1 = self.tools.get_spectral_mean(band1_list, array)
        array_b2 = self.tools.get_spectral_mean(band2_list, array)
        array_ndi = (array_b1-array_b2)/(array_b1+array_b2)

        metadata['bands'] = 1
        self.tools.del_meta_item(metadata, 'wavelength')
        self.tools.del_meta_item(metadata, 'band names')
        hist_str = (" -> hs_process.band_math_ndi[<"
                    "label: 'b1?' value:{0}; "
                    "label: 'b2?' value:{1}; "
                    "label: 'list_range?' value:{2}>]"
                    "".format(b1, b2, list_range))
        metadata['history'] += hist_str
        metadata['samples'] = array_ndi.shape[1]
        metadata['lines'] = array_ndi.shape[0]
        return array_ndi, metadata

    def load_spyfile(self, spyfile):
        '''
        Loads a `SpyFile` (Spectral Python object) for data access and/or
        manipulation by the `hstools` class.

        Parameters:
            spyfile (`SpyFile` object): The datacube being accessed and/or
                manipulated.
        '''
        self.spyfile = spyfile
        self.tools = hstools(spyfile)
        try:
            self.spy_ul_e_srs = float(self.spyfile.metadata['map info'][3])
            self.spy_ul_n_srs = float(self.spyfile.metadata['map info'][4])
            self.spy_ps_e = float(self.spyfile.metadata['map info'][5])
            self.spy_ps_n = float(self.spyfile.metadata['map info'][6])
        except KeyError as e:
            print('Map information was not able to be loaded from the '
                  '`SpyFile`. Please be sure the metadata contains the "map '
                  'info" tag with accurate geometric information.\n')
            self.spy_ul_e_srs = None
            self.spy_ul_n_srs = None
            self.spy_ps_e = None
            self.spy_ps_n = None

    def veg_spectra(self, array_veg, thresh=None, percentile=0.5, side='lower',
                    spyfile=None):
        '''
        Calculates the average spectra across vegetation pixels

        Parameters:
            array_veg (`numpy.ndarray`): a single-band image array, presumably
                that discriminates vegetation pixels from other pixels such as
                soil, shadow, etc.
            thresh (`float`): The value for which to base the threshold
                (default: `None`).
            percentile (`float`): The percentile of pixels to mask; if
                `percentile`=0.95 and `side`='lower', the lowest 95% of pixels
                will be masked prior to calculating the mean spectra across
                pixels (default: 0.5).
            side (`str`): The side of the threshold for which to apply the
                mask. Must be either 'lower' or 'upper'; if 'lower', everything
                below the threshold will be masked (default: 'lower').
        '''
        if spyfile is None:
            spyfile = self.spyfile
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
        metadata = self.tools.spyfile.metadata

        mask_array = self.tools.mask_array(array_veg, thresh=thresh, side=side)
        mask_array_3d = np.empty(spyfile.shape)
        for band in range(spyfile.nbands):
            mask_array_3d[:, :, band] = mask_array.mask
        array_veg_masked = np.ma.masked_array(array_veg, mask=mask_array_3d)
        veg_spectra = np.mean(array_veg_masked, axis=(0, 1))
        return veg_spectra, metadata
