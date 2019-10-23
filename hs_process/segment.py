# -*- coding: utf-8 -*-
from spectral import kmeans
import numpy as np
import pandas as pd
import spectral.io.spyfile as SpyFile
import seaborn as sns
from matplotlib import pyplot as plt

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

    def _check_bands_wls(self, wl, b, n):
        '''
        Checks to be sure there is a valid wavelength or band to be used
        '''
        msg1 = ('At least one of either `wl{0}` or `b{0}` must be passed. '
                'Please check your inputs.\n'.format(n))
        if wl is None and b is None:
            raise ValueError(msg1)
        msg2 = ('Both `wl{0} and b{0} were passed, but only one can be used '
                'to deterimine which image band to use; the wavelength band '
                '(`wl{0}`={1}) will be used for the band math operation.\n'
                ''.format(n, wl))
        if wl is not None and b is not None:
            print(msg2)
        elif wl is not None and b is None:  # use wl
            pass
        elif wl is None and b is not None:  # get wl
            wl = self.tools.get_wavelength(b)
        return wl

    def _get_band_list(self, wl_list, list_range):
        '''
        Determines how a list of wavelengths should be consolidated, if at all.
        '''
        if isinstance(wl_list, list) and list_range is True:
            msg = ('When using a `list_range`, please be sure each passed '
                   '"band" is a list of exactly two wavelength values.\n')
            assert len(wl_list) == 2, msg
            band_list = self.tools.get_band_range(wl_list, index=False)
        elif isinstance(wl_list, list) and list_range is False:
            band_list = []
            for b_i in wl_list:
                b = self.tools.get_band(b_i)
                band_list.append(b)
        else:  # just a single band; disregards `list_range`
            b = self.tools.get_band(wl_list)
            band_list = [b]
        return band_list

    def _get_wavelength_list(self, band_list_in, list_range):
        '''
        Determines how a list of bands should be consolidated, if at all.
        '''
        if isinstance(band_list_in, list) and list_range is True:
            msg = ('When using a `list_range`, please be sure each passed '
                   '"band" is a list of exactly two wavelength values.\n')
            assert len(band_list_in) == 2, msg
            wl_list = self.tools.get_band_range(band_list_in, index=False)
        elif isinstance(band_list_in, list) and list_range is False:
            wl_list = []
            for b_i in band_list_in:
                wl = self.tools.get_wavelength(b_i)
                wl_list.append(wl)
        else:  # just a single band; disregards `list_range`
            wl = self.tools.get_wavelength(band_list_in)
            wl_list = [wl]
        return wl_list

    def band_math_derivative(self, wl1=None, wl2=None, wl3=None,
                             b1=None, b2=None, b3=None,
                             spyfile=None, list_range=True, print_out=True):
        '''
        Calculates a derivative-type spectral index from two input bands
        and/or wavelengths. Bands/wavelengths can be input as two individual
        bands, two sets of bands (i.e., list of bands), or range of bands
        (i.e., list of two bands indicating the lower and upper range).

        der_index = (wl1 - wl2) / (wl2 - wl3)
        Parameters:
            wl1 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                derivative index; if `list`, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: `None`).
            wl2 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the second parameter of the
                derivative index; if `list`, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: `None`).
            b1 (`int`, `float`, or `list`): the band (or set of bands) to be
                used as the first parameter of the derivative index;
                if `list`, then consolidates all bands between two band values
                by calculating the mean pixel value across all bands in that
                range (default: `None`).
            b2 (`int`, `float`, or `list`): the band (or set of bands) to be
                used as the second parameter of the derivative
                index; if `list`, then consolidates all bands between two band
                values by calculating the mean pixel value across all bands in
                that range (default: `None`).
            spyfile (`SpyFile` object or `numpy.ndarray`): The datacube to
                crop; if `numpy.ndarray` or `None`, loads band information from
                `self.spyfile` (default: `None`).
            list_range (`bool`): Whether bands/wavelengths passed as a list is
                interpreted as a range of bands (`True`) or for each individual
                band in the list (`False`). If `list_range` is `True`,
                `b1`/`wl1` and `b2`/`wl2` should be lists with two items, and
                all bands/wavelegths between the two values will be used
                (default: `True`).
            print_out (`bool`): Whether to print out the actual bands and
                wavelengths being used in the NDI calculation (default:
                `True`).
            '''
        wl1 = self._check_bands_wls(wl1, b1, 1)
        wl2 = self._check_bands_wls(wl2, b2, 2)
        wl3 = self._check_bands_wls(wl3, b3, 3)
        # Now, band input is converted to wavelength input and this can be used

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

        band1_list = self._get_band_list(wl1, list_range)
        band2_list = self._get_band_list(wl2, list_range)
        band3_list = self._get_band_list(wl3, list_range)

        if print_out is True:
            wl1_list = self._get_wavelength_list(band1_list, list_range=False)
            wl2_list = self._get_wavelength_list(band2_list, list_range=False)
            wl3_list = self._get_wavelength_list(band3_list, list_range=False)
            print('\nBands used (`b1`): {0}'.format(band1_list))
            print('Bands used (`b2`): {0}'.format(band2_list))
            print('Bands used (`b3`): {0}'.format(band3_list))
            print('\nWavelengths used (`b1`): {0}'.format(wl1_list))
            print('Wavelengths used (`b2`): {0}'.format(wl2_list))
            print('Wavelengths used (`b3`): {0}\n'.format(wl3_list))

        array_b1 = self.tools.get_spectral_mean(band1_list, array)
        array_b2 = self.tools.get_spectral_mean(band2_list, array)
        array_b3 = self.tools.get_spectral_mean(band3_list, array)
        array_der = (array_b1-array_b2)/(array_b2-array_b3)

        metadata['bands'] = 1
        self.tools.del_meta_item(metadata, 'wavelength')
        self.tools.del_meta_item(metadata, 'band names')
        hist_str = (" -> hs_process.band_math_ndi[<"
                    "label: 'wl1?' value:{0}; "
                    "label: 'wl2?' value:{1}; "
                    "label: 'wl3?' value:{2}; "
                    "label: 'list_range?' value:{3}>]"
                    "".format(wl1_list, wl2_list, wl3_list, list_range))
        metadata['history'] += hist_str
        metadata['samples'] = array_der.shape[1]
        metadata['lines'] = array_der.shape[0]
        return array_der, metadata

    def band_math_ratio(self, wl1=None, wl2=None, b1=None, b2=None,
                        spyfile=None, list_range=True, print_out=True):
        '''
        Calculates a simple ratio spectral index from two input band and/or
        wavelengths. Bands/wavelengths can be input as two individual bands,
        two sets of bands (i.e., list of bands), or a range of bands (i.e.,
        list of two bands indicating the lower and upper range).

        Parameters:
            wl1 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                normalized difference index; if `list`, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: `None`).
            wl2 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the second parameter of the
                normalized difference index; if `list`, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: `None`).
            b1 (`int`, `float`, or `list`): the band (or set of bands) to be
                used as the first parameter (numerator) of the ratio index; if
                `list`, then consolidates all bands between two band values by
                calculating the mean pixel value across all bands in that
                range (default: `None`).
            b2 (`int`, `float`, or `list`): the bands (or set of bands) to be\
                used as the second parameter (denominator) of the ratio index;
                if `list`, then consolidates all bands between two bands values
                by calculating the mean pixel value across all bands in that
                range (default: `None`).
            spyfile (`SpyFile` object or `numpy.ndarray`): The datacube to
                crop; if `numpy.ndarray` or `None`, loads band information from
                `self.spyfile` (default: `None`).
            list_range (`bool`): Whether a band passed as a list is interpreted as a
                range of bands (`True`) or for each individual band in the
                list (`False`). If `list_range` is `True`, `b1`/`wl1` and
                `b2`/`wl2` should be lists with two items, and all
                bands/wavelegths between the two values will be used (default:
                `True`).
            print_out (`bool`): Whether to print out the actual bands and
                wavelengths being used in the NDI calculation (default:
                `True`).
        '''
        wl1 = self._check_bands_wls(wl1, b1, 1)
        wl2 = self._check_bands_wls(wl2, b2, 2)
        # Now, band input is converted to wavelength input and this can be used

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

        band1_list = self._get_band_list(wl1, list_range)
        band2_list = self._get_band_list(wl2, list_range)

        if print_out is True:
            wl1_list = self._get_wavelength_list(band1_list, list_range=False)
            wl2_list = self._get_wavelength_list(band2_list, list_range=False)
            print('\nBands used (`b1`): {0}'.format(band1_list))
            print('Bands used (`b2`): {0}'.format(band2_list))
            print('\nWavelengths used (`b1`): {0}'.format(wl1_list))
            print('Wavelengths used (`b2`): {0}\n'.format(wl2_list))
            print('({0:.0f}/{1:.0f})'.format(np.mean(wl1_list),
                                             np.mean(wl2_list)))

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

    def band_math_ndi(self, wl1=None, wl2=None, b1=None, b2=None, spyfile=None,
                      list_range=True, print_out=True):
        '''
        Calculates a normalized difference spectral index from two input bands
        and/or wavelengths. Bands/wavelengths can be input as two individual
        bands, two sets of bands (i.e., list of bands), or range of bands
        (i.e., list of two bands indicating the lower and upper range).

        Parameters:
            wl1 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                normalized difference index; if `list`, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: `None`).
            wl2 (`int`, `float`, or `list`): the wavelength (or set of
                wavelengths) to be used as the second parameter of the
                normalized difference index; if `list`, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: `None`).
            b1 (`int`, `float`, or `list`): the band (or set of bands) to be
                used as the first parameter of the normalized difference index;
                if `list`, then consolidates all bands between two band values
                by calculating the mean pixel value across all bands in that
                range (default: `None`).
            b2 (`int`, `float`, or `list`): the band (or set of bands) to be
                used as the second parameter of the normalized difference
                index; if `list`, then consolidates all bands between two band
                values by calculating the mean pixel value across all bands in
                that range (default: `None`).
            spyfile (`SpyFile` object or `numpy.ndarray`): The datacube to
                crop; if `numpy.ndarray` or `None`, loads band information from
                `self.spyfile` (default: `None`).
            list_range (`bool`): Whether bands/wavelengths passed as a list is
                interpreted as a range of bands (`True`) or for each individual
                band in the list (`False`). If `list_range` is `True`,
                `b1`/`wl1` and `b2`/`wl2` should be lists with two items, and
                all bands/wavelegths between the two values will be used
                (default: `True`).
            print_out (`bool`): Whether to print out the actual bands and
                wavelengths being used in the NDI calculation (default:
                `True`).
        '''
        wl1 = self._check_bands_wls(wl1, b1, 1)
        wl2 = self._check_bands_wls(wl2, b2, 2)
        # Now, band input is converted to wavelength input and this can be used

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

        band1_list = self._get_band_list(wl1, list_range)
        band2_list = self._get_band_list(wl2, list_range)
        if print_out is True:
            wl1_list = self._get_wavelength_list(band1_list, list_range=False)
            wl2_list = self._get_wavelength_list(band2_list, list_range=False)
            print('\nBands used (`b1`): {0}'.format(band1_list))
            print('Bands used (`b2`): {0}'.format(band2_list))
            print('\nWavelengths used (`b1`): {0}'.format(wl1_list))
            print('Wavelengths used (`b2`): {0}\n'.format(wl2_list))

        array_b1 = self.tools.get_spectral_mean(band1_list, array)
        array_b2 = self.tools.get_spectral_mean(band2_list, array)
        array_ndi = (array_b1-array_b2)/(array_b1+array_b2)

        metadata['bands'] = 1
        self.tools.del_meta_item(metadata, 'wavelength')
        self.tools.del_meta_item(metadata, 'band names')
        hist_str = (" -> hs_process.band_math_ndi[<"
                    "label: 'wl1?' value:{0}; "
                    "label: 'wl2?' value:{1}; "
                    "label: 'list_range?' value:{2}>]"
                    "".format(wl1_list, wl2_list, list_range))
        metadata['history'] += hist_str
        metadata['samples'] = array_ndi.shape[1]
        metadata['lines'] = array_ndi.shape[0]
        return array_ndi, metadata

    def band_math_mcari2(self, wl1=None, wl2=None, wl3=None, b1=None, b2=None,
                         b3=None, spyfile=None, list_range=True,
                         print_out=True):
        '''
        Calculates the MCARI2 spectral index
            '''
        wl1 = self._check_bands_wls(wl1, b1, 1)
        wl2 = self._check_bands_wls(wl2, b2, 2)
        wl3 = self._check_bands_wls(wl3, b3, 3)
        # Now, band input is converted to wavelength input and this can be used

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

        band1_list = self._get_band_list(wl1, list_range)
        band2_list = self._get_band_list(wl2, list_range)
        band3_list = self._get_band_list(wl3, list_range)

        if print_out is True:
            wl1_list = self._get_wavelength_list(band1_list, list_range=False)
            wl2_list = self._get_wavelength_list(band2_list, list_range=False)
            wl3_list = self._get_wavelength_list(band3_list, list_range=False)
            print('\nBands used (`b1`): {0}'.format(band1_list))
            print('Bands used (`b2`): {0}'.format(band2_list))
            print('Bands used (`b3`): {0}'.format(band3_list))
            print('\nWavelengths used (`b1`): {0}'.format(wl1_list))
            print('Wavelengths used (`b2`): {0}'.format(wl2_list))
            print('Wavelengths used (`b3`): {0}\n'.format(wl3_list))

        array_b1 = self.tools.get_spectral_mean(band1_list, array)
        array_b2 = self.tools.get_spectral_mean(band2_list, array)
        array_b3 = self.tools.get_spectral_mean(band3_list, array)
#        array_der = (array_b1-array_b2)/(array_b2-array_b3)
        array_mcari2 = ((1.5 * (2.5 * (array_b1 - array_b2) - 1.3 * (array_b1 - array_b3))) /
                        np.sqrt((2 * array_b1 + 1)**2 - (6 * array_b1 - 5 * np.sqrt(array_b2)) - 0.5))

        metadata['bands'] = 1
        self.tools.del_meta_item(metadata, 'wavelength')
        self.tools.del_meta_item(metadata, 'band names')
        hist_str = (" -> hs_process.band_math_mcari2[<"
                    "label: 'wl1?' value:{0}; "
                    "label: 'wl2?' value:{1}; "
                    "label: 'wl3?' value:{2}; "
                    "label: 'list_range?' value:{3}>]"
                    "".format(wl1_list, wl2_list, wl3_list, list_range))
        metadata['history'] += hist_str
        metadata['samples'] = array_mcari2.shape[1]
        metadata['lines'] = array_mcari2.shape[0]
        return array_mcari2, metadata

    def kmeans(self, n_classes=3, max_iter=100, spyfile=None):
        '''
        If there are more soil pixels than vegetation pixels, will vegetation be
        class 0 instead of class 2?

        Parameters:
            n_classes (`int`): number of classes (default: 3).
            max_iter (`int`): maximum iterations before terminating process
                (default: 100).
            spyfile (`SpyFile` object or `numpy.ndarray`): The datacube to
                crop; if `numpy.ndarray` or `None`, loads band information from
                `self.spyfile` (default: `None`).
        '''
        if spyfile is None:
            spyfile = self.spyfile
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
        elif isinstance(spyfile, np.ndarray):
            spyfile = self.spyfile
        metadata = self.tools.spyfile.metadata

        array_class, c = kmeans(spyfile, n_classes, max_iter)
        df_class_spec = pd.DataFrame(c.transpose())

#        nir_b = self.tools.get_band(760)
#        re_b = self.tools.get_band(720)
#        red_b = self.tools.get_band(680)
#        green_b = self.tools.get_band(555)
#        nir = df_class_spec.iloc[nir_b]
#        re = df_class_spec.iloc[re_b]
#        red = df_class_spec.iloc[red_b]
#        green = df_class_spec.iloc[green_b]
#        df_ndvi = (nir-red)/(nir+red)
#        df_gndvi = (nir-green)/(nir+green)
#        df_rendvi = (nir-re)/(nir+re)

        metadata['bands'] = 1
        self.tools.del_meta_item(metadata, 'wavelength')
        self.tools.del_meta_item(metadata, 'band names')
        hist_str = (" -> hs_process.segment.kmeans[<"
                    "label: 'n_classes?' value:{0}; "
                    "label: 'max_iter?' value:{1}>]"
                    "".format(n_classes, max_iter))
        metadata['history'] += hist_str
        metadata['samples'] = array_class.shape[1]
        metadata['lines'] = array_class.shape[0]
        return array_class, df_class_spec, metadata

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

    def veg_spectra(self, array_veg, thresh=None, percentile=None,
                    side='lower', spyfile=None):
        '''
        Calculates the average spectra across vegetation pixels

        Parameters:
            array_veg (`numpy.ndarray`): a single-band image array, presumably
                that discriminates vegetation pixels from other pixels such as
                soil, shadow, etc.
            thresh (`float`): The value for which to base the threshold
                (default: `None`).
            percentile (`float` or `int`): The percentile of pixels to mask; if
                `percentile`=95 and `side`='lower', the lowest 95% of pixels
                will be masked prior to calculating the mean spectra across
                pixels (default: `None`; range: 0-100).
            side (`str`): The side of the threshold or percentile for which to
                apply the mask. Must be either 'lower' or 'upper'; if 'lower',
                everything below the threshold/percentile will be masked
                (default: 'lower').
        '''
        if spyfile is None:
            spyfile = self.spyfile
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)

#        if remove_shadow is True:
#            shadow_mask, metadata = self.tools.mask_shadow(
#                    shadow_pctl=shadow_pctl, show_histogram=True,
#                    spyfile=spyfile)
#            array_veg = np.ma.array(array_veg, mask=shadow_mask)
#        else:
#            if not isinstance(array_veg, np.ma.core.MaskedArray):
#                array_veg = np.ma.array(array_veg, mask=False)
#            metadata = self.spyfile.metadata

        mask_array, metadata = self.tools.mask_array(
                array_veg, self.spyfile.metadata, thresh=thresh,
                percentile=percentile, side=side)

        mask_array_3d = np.empty(spyfile.shape)
        for band in range(spyfile.nbands):
            mask_array_3d[:, :, band] = mask_array.mask
        datacube_masked = np.ma.masked_array(spyfile.load(),
                                             mask=mask_array_3d)
        spec_mean = np.nanmean(datacube_masked, axis=(0, 1))
        spec_std = np.nanstd(datacube_masked, axis=(0, 1))
#        a = spec_mean.reshape(len(spec_mean))
        spec_mean = pd.Series(spec_mean)
        spec_std = pd.Series(spec_std)

        return spec_mean, spec_std, datacube_masked, metadata

    def mask_datacube(self, mask, spyfile=None):
        '''
        Applies `mask` to `spyfile`, then returns the datcube (as a np.array)
        and the mean spectra

        Parameters:
            mask (`numpy.ndarray`): the mask to apply to `spyfile`; if `mask`
                does not have similar dimensions to `spyfile`, the first band
                (i.e., first two dimensions) of `mask` will be repeated n times
                to match the number of bands of `spyfile`.
            spyfile (`SpyFile` object): The datacube being accessed and/or
                manipulated.
        '''
        if spyfile is None:
            spyfile = self.spyfile
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)

        if isinstance(mask, np.ma.masked_array):
            mask = mask.mask
        if mask.shape != spyfile.shape:
            mask_1d = mask.copy()
            mask = np.empty(spyfile.shape)
            for band in range(spyfile.nbands):
                mask[:, :, band] = mask_1d

        datacube_masked = np.ma.masked_array(spyfile.load(), mask=mask)
        spec_mean = np.nanmean(datacube_masked, axis=(0, 1))
        spec_std = np.nanstd(datacube_masked, axis=(0, 1))
        spec_mean = pd.Series(spec_mean)
        spec_std = pd.Series(spec_std)
        return spec_mean, spec_std, datacube_masked