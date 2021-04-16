# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import spectral.io.spyfile as SpyFile
import seaborn as sns
from matplotlib import pyplot as plt

from hs_process.utilities import defaults
from hs_process.utilities import hstools


class segment(object):
    '''
    Class for aiding in the segmentation and/or masking of image
    data to filter out pixels that are of least interest.
    '''
    def __init__(self, spyfile):
        '''
        spyfile (``SpyFile`` object): The Spectral Python datacube to manipulate.
        '''
        self.spyfile = spyfile
        self.tools = hstools(spyfile)

        self.spy_ps_e = None
        self.spy_ps_n = None
        self.spy_ul_e_srs = None
        self.spy_ul_n_srs = None

        self.defaults = defaults
        self.load_spyfile(spyfile)

    def _check_bands_wls(self, wl, b, n=''):
        '''
        Checks to be sure there is a valid wavelength or band to be used
        '''
        msg1 = ('At least one of either ``wl{0}`` or ``b{0}`` must be passed. '
                'Please check your inputs.\n'.format(n))
        if wl is None and b is None:
            raise ValueError(msg1)
        msg2 = ('Both ``wl{0} and b{0} were passed, but only one can be used '
                'to deterimine which image band to use; the wavelength band '
                '(``wl{0}``={1}) will be used for the band math operation.\n'
                ''.format(n, wl))
        if wl is not None and b is not None:
            print(msg2)
        elif wl is not None and b is None:  # use wl
            pass
        elif wl is None and b is not None:  # get wl
            wl = self.tools.get_wavelength(b)
        return wl

    def _check_classes(self, array_class, n_pix):
        '''
        Checks that enough classes were assigned
        '''
        unique_classes, counts = np.unique(array_class, return_counts=True)
        unique_classes_mod = list(unique_classes.copy())
        counts_mod = list(counts.copy())
        for idx, (unique, count) in enumerate(zip(unique_classes, counts)):
            if count < n_pix:
                unique_classes_mod.pop(unique)
                counts_mod.pop(idx)
        return unique_classes_mod, counts_mod

    def _get_band_list(self, wl_list, list_range):
        '''
        Determines how a list of wavelengths should be consolidated, if at all.

        If wl_list is a list with only a single band, ``list_range`` will be
        ignored and that single band will be used.
        '''
        if isinstance(wl_list, list) and list_range is True:
            msg = ('When using a ``list_range``, please be sure each passed '
                   '"band" is a list of exactly two wavelength values.\n')
            # assert len(wl_list) <= 2, msg
            if len(wl_list) == 1:
                band_list = [self.tools.get_band(wl_list[0])]
            elif len(wl_list) == 2:
                band_list = self.tools.get_band_range(wl_list, index=False)
            else:
                raise ValueError(msg)
        elif isinstance(wl_list, list) and list_range is False:
            band_list = []
            for b_i in wl_list:
                b = self.tools.get_band(b_i)
                band_list.append(b)
        else:  # just a single band; disregards ``list_range``
            band_list = [self.tools.get_band(wl_list)]
        return band_list

    def _get_wavelength_list(self, band_list_in, list_range):
        '''
        Determines how a list of bands should be consolidated, if at all.
        '''
        if isinstance(band_list_in, list) and list_range is True:
            msg = ('When using a ``list_range``, please be sure each passed '
                   '"band" is a list of exactly two wavelength values.\n')
            assert len(band_list_in) == 2, msg
            wl_list = self.tools.get_band_range(band_list_in, index=False)
        elif isinstance(band_list_in, list) and list_range is False:
            wl_list = []
            for b_i in band_list_in:
                wl = self.tools.get_wavelength(b_i)
                wl_list.append(wl)
        else:  # just a single band; disregards ``list_range``
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

        Definition:
            array_der = (``wl1`` - ``wl2``) / (``wl2`` - ``wl3``)

        Parameters:
            wl1 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                derivative index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            wl2 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the second parameter of the
                derivative index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            wl3 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the third parameter of the
                derivative index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            b1 (``int``, ``float``, or ``list``): the band (or set of bands) to
                be used as the first parameter of the derivative index;
                if ``list``, then consolidates all bands between two band
                values by calculating the mean pixel value across all bands in
                that range (default: ``None``).
            b2 (``int``, ``float``, or ``list``): the band (or set of bands) to
                be used as the second parameter of the derivative index; if
                ``list``, then consolidates all bands between two band values
                by calculating the mean pixel value across all bands in that
                range (default: ``None``).
            b3 (``int``, ``float``, or ``list``): the band (or set of bands) to
                be used as the third parameter of the derivative index; if
                ``list``, then consolidates all bands between two band values
                by calculating the mean pixel value across all bands in that
                range (default: ``None``).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The datacube to
                process; if ``numpy.ndarray`` or ``None``, loads band information
                from ``self.spyfile`` (default: ``None``).
            list_range (``bool``): Whether bands/wavelengths passed as a list
                is interpreted as a range of bands (``True``) or for each
                individual band in the list (``False``). If ``list_range`` is
                ``True``, ``b1``/``wl1`` and ``b2``/``wl2`` should be lists
                with two items, and all bands/wavelegths between the two values
                will be used (default: ``True``).
            print_out (``bool``): Whether to print out the actual bands and
                wavelengths being used in the NDI calculation (default:
                ``True``).

        Returns:
            2-element ``tuple`` containing

            - **array_der** (``numpy.ndarray``): Derivative band math array.
            - **metadata** (``dict``): Modified metadata describing the
              derivative array (``array_der``).

        Example:
            Load ``hsio`` and ``segment`` modules

            >>> import numpy as np
            >>> import os
            >>> from hs_process import hsio
            >>> from hs_process import segment
            >>> data_dir = r'F:\\nigo0024\Documents\hs_process_demo'
            >>> fname_in = os.path.join(data_dir, 'Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr')
            >>> io = hsio(fname_in)
            >>> my_segment = segment(io.spyfile)

            Calculate the MERIS Terrestrial Chlorophyll Index (MTCI; Dash and
            Curran, 2004) via ``segment.band_math_derivative``

            >>> array_mtci, metadata = my_segment.band_math_derivative(wl1=754, wl2=709, wl3=681, spyfile=io.spyfile)
            Band 1: [176]  Band 2: [154]  Band 3: [141]
            Wavelength 1: [753.84]  Wavelength 2: [708.6784]  Wavelength 3: [681.992]

            >>> array_mtci.shape
            (617, 1300)
            >>> np.nanmean(array_mtci)
            9.401104

            Show MTCI image via ``hsio.show_img``

            >>> io.show_img(array_mtci, vmin=-2, vmax=15)

            .. image:: ../img/segment/mtci.png
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
        wl1_list = self._get_wavelength_list(band1_list, list_range=False)
        wl2_list = self._get_wavelength_list(band2_list, list_range=False)
        wl3_list = self._get_wavelength_list(band3_list, list_range=False)
        if print_out is True:
            print('Band 1: {0}  Band 2: {1}  Band 3: {2}'
                  ''.format(band1_list, band2_list, band3_list))
            print('Wavelength 1: {0}  Wavelength 2: {1}  Wavelength 3: {2}'
                  ''.format(wl1_list, wl2_list, wl3_list))

        array_b1 = self.tools.get_spectral_mean(band1_list, array)
        array_b2 = self.tools.get_spectral_mean(band2_list, array)
        array_b3 = self.tools.get_spectral_mean(band3_list, array)
        with np.errstate(invalid='ignore', divide='ignore'):
            array_der = (array_b1-array_b2)/(array_b2-array_b3)
        array_der[array_der == 0] = np.nan

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

    def band_math_mcari2(self, wl1=None, wl2=None, wl3=None, b1=None, b2=None,
                         b3=None, spyfile=None, list_range=True,
                         print_out=True):
        '''
        Calculates the MCARI2 (Modified Chlorophyll Absorption Ratio Index
        Improved; Haboudane et al., 2004) spectral index from three input bands
        and/or wavelengths. Bands/wavelengths can be input as two individual
        bands, two sets of bands (i.e., list of bands), or range of bands
        (i.e., list of two bands indicating the lower and upper range).

        Definition:
            array_mcari2 = ((1.5 * (2.5 * (``wl1`` - ``wl2``) - 1.3 * (``wl1`` - ``wl3``))) / np.sqrt((2 * ``wl1`` + 1)**2 - (6 * ``wl1`` - 5 * np.sqrt(``wl2``)) - 0.5))

        Parameters:
            wl1 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                MCARI2 index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            wl2 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the second parameter of the
                MCARI2 index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            wl3 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the third parameter of the
                MCARI2 index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            b1 (``int``, ``float``, or ``list``): the band (or set of bands) to
                be used as the first parameter of the MCARI2 index;
                if ``list``, then consolidates all bands between two band
                values by calculating the mean pixel value across all bands in
                that range (default: ``None``).
            b2 (``int``, ``float``, or ``list``): the band (or set of bands) to
                be used as the second parameter of the MCARI2 index; if
                ``list``, then consolidates all bands between two band values
                by calculating the mean pixel value across all bands in that
                range (default: ``None``).
            b3 (``int``, ``float``, or ``list``): the band (or set of bands) to
                be used as the third parameter of the MCARI2 index; if
                ``list``, then consolidates all bands between two band values
                by calculating the mean pixel value across all bands in that
                range (default: ``None``).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The datacube to
                process; if ``numpy.ndarray`` or ``None``, loads band information
                from ``self.spyfile`` (default: ``None``).
            list_range (``bool``): Whether bands/wavelengths passed as a list
                is interpreted as a range of bands (``True``) or for each
                individual band in the list (``False``). If ``list_range`` is
                ``True``, ``b1``/``wl1`` and ``b2``/``wl2`` should be lists
                with two items, and all bands/wavelegths between the two values
                will be used (default: ``True``).
            print_out (``bool``): Whether to print out the actual bands and
                wavelengths being used in the NDI calculation (default:
                ``True``).

        Returns:
            2-element ``tuple`` containing

            - **array_mcari2** (``numpy.ndarray``): MCARI2 spectral index band
              math array.
            - **metadata** (``dict``): Modified metadata describing the
              MCARI2 index array (``array_mcari2``).

        Example:
            Load ``hsio`` and ``segment`` modules

            >>> import numpy as np
            >>> import os
            >>> from hs_process import hsio
            >>> from hs_process import segment
            >>> data_dir = r'F:\\nigo0024\Documents\hs_process_demo'
            >>> fname_in = os.path.join(data_dir, 'Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr')
            >>> io = hsio(fname_in)
            >>> my_segment = segment(io.spyfile)

            Calculate the MCARI2 spectral index (Haboudane et al., 2004) via
            ``segment.band_math_mcari2``

            >>> array_mcari2, metadata = my_segment.band_math_mcari2(wl1=800, wl2=670, wl3=550, spyfile=io.spyfile)
            Band 1: [198]  Band 2: [135]  Band 3: [77]
            Wavelength 1: [799.0016]  Wavelength 2: [669.6752]  Wavelength 3: [550.6128]

            >>> np.nanmean(array_mcari2)
            0.57376945

            Show MCARI2 image via ``hsio.show_img``

            >>> io.show_img(array_mcari2)

            .. image:: ../img/segment/mcari2.png
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
        wl1_list = self._get_wavelength_list(band1_list, list_range=False)
        wl2_list = self._get_wavelength_list(band2_list, list_range=False)
        wl3_list = self._get_wavelength_list(band3_list, list_range=False)
        if print_out is True:
            print('Band 1: {0}  Band 2: {1}  Band 3: {2}'
                  ''.format(band1_list, band2_list, band3_list))
            print('Wavelength 1: {0}  Wavelength 2: {1}  Wavelength 3: {2}'
                  ''.format(wl1_list, wl2_list, wl3_list))

        array_b1 = self.tools.get_spectral_mean(band1_list, array)
        array_b2 = self.tools.get_spectral_mean(band2_list, array)
        array_b3 = self.tools.get_spectral_mean(band3_list, array)
#        array_der = (array_b1-array_b2)/(array_b2-array_b3)
        with np.errstate(invalid='ignore', divide='ignore'):
            array_mcari2 = ((1.5 * (2.5 * (array_b1 - array_b2) - 1.3 * (array_b1 - array_b3))) /
                            np.sqrt((2 * array_b1 + 1)**2 - (6 * array_b1 - 5 * np.sqrt(array_b2)) - 0.5))
        array_mcari2[array_mcari2 == 0] = np.nan

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

    def band_math_ndi(self, wl1=None, wl2=None, b1=None, b2=None, spyfile=None,
                      list_range=True, print_out=True):
        '''
        Calculates a normalized difference spectral index from two input bands
        and/or wavelengths. Bands/wavelengths can be input as two individual
        bands, two sets of bands (i.e., list of bands), or range of bands
        (i.e., list of two bands indicating the lower and upper range).

        Definition:
            array_ndi = (``wl1`` - ``wl2``) / (``wl1`` + ``wl2``)

        Parameters:
            wl1 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                normalized difference index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            wl2 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the second parameter of the
                normalized difference index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            b1 (``int``, ``float``, or ``list``): the band (or set of bands) to be
                used as the first parameter of the normalized difference index;
                if ``list``, then consolidates all bands between two band values
                by calculating the mean pixel value across all bands in that
                range (default: ``None``).
            b2 (``int``, ``float``, or ``list``): the band (or set of bands) to be
                used as the second parameter of the normalized difference
                index; if ``list``, then consolidates all bands between two band
                values by calculating the mean pixel value across all bands in
                that range (default: ``None``).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The datacube to
                process; if ``numpy.ndarray`` or ``None``, loads band information from
                ``self.spyfile`` (default: ``None``).
            list_range (``bool``): Whether bands/wavelengths passed as a list is
                interpreted as a range of bands (``True``) or for each individual
                band in the list (``False``). If ``list_range`` is ``True``,
                ``b1``/``wl1`` and ``b2``/``wl2`` should be lists with two items, and
                all bands/wavelegths between the two values will be used
                (default: ``True``).
            print_out (``bool``): Whether to print out the actual bands and
                wavelengths being used in the NDI calculation (default:
                ``True``).

        Returns:
            2-element ``tuple`` containing

            - **array_ndi** (``numpy.ndarray``): Normalized difference band
              math array.
            - **metadata** (``dict``): Modified metadata describing the
              normalized difference array (``array_ndi``).

        Example:
            Load ``hsio`` and ``segment`` modules

            >>> import numpy as np
            >>> import os
            >>> from hs_process import hsio
            >>> from hs_process import segment
            >>> data_dir = r'F:\\nigo0024\Documents\hs_process_demo'
            >>> fname_in = os.path.join(data_dir, 'Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr')
            >>> io = hsio(fname_in)
            >>> my_segment = segment(io.spyfile)

            Calculate the Normalized difference vegetation index using 10 nm
            bands centered at 800 nm and 680 nm via ``segment.band_math_ndi``

            >>> array_ndvi, metadata = my_segment.band_math_ndi(wl1=[795, 805], wl2=[675, 685], spyfile=io.spyfile)
            Effective NDI: (800 - 680) / 800 + 680)

            >>> np.nanmean(array_ndvi)
            0.8184888

            Show NDVI image via ``hsio.show_img``

            >>> io.show_img(array_ndvi)

            .. image:: ../img/segment/ndvi.png
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
        wl1_list = self._get_wavelength_list(band1_list, list_range=False)
        wl2_list = self._get_wavelength_list(band2_list, list_range=False)
        if print_out is True:
            print('Effective NDI: ({0:.0f} - {1:.0f}) / {0:.0f} + {1:.0f})'
                  ''.format(np.mean(wl1_list), np.mean(wl2_list)))

        array_b1 = self.tools.get_spectral_mean(band1_list, array)
        array_b2 = self.tools.get_spectral_mean(band2_list, array)
        with np.errstate(invalid='ignore', divide='ignore'):
            array_ndi = (array_b1-array_b2)/(array_b1+array_b2)
        array_ndi[array_ndi == 0] = np.nan

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

    def band_math_ratio(self, wl1=None, wl2=None, b1=None, b2=None,
                        spyfile=None, list_range=True, print_out=True):
        '''
        Calculates a simple ratio spectral index from two input band and/or
        wavelengths. Bands/wavelengths can be input as two individual bands,
        two sets of bands (i.e., list of bands), or a range of bands (i.e.,
        list of two bands indicating the lower and upper range).

        Definition:
            array_ratio = (``wl1``/``wl2``)

        Parameters:
            wl1 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the first parameter of the
                ratio index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            wl2 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to be used as the second parameter of the
                ratio index; if ``list``, then consolidates all
                bands between two wavelength values by calculating the mean
                pixel value across all bands in that range (default: ``None``).
            b1 (``int``, ``float``, or ``list``): the band (or set of bands) to
                be used as the first parameter (numerator) of the ratio index;
                if ``list``, then consolidates all bands between two band
                values by calculating the mean pixel value across all bands in
                that range (default: ``None``).
            b2 (``int``, ``float``, or ``list``): the bands (or set of bands)
                to be used as the second parameter (denominator) of the ratio
                index; if ``list``, then consolidates all bands between two
                bands values by calculating the mean pixel value across all
                bands in that range (default: ``None``).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The datacube to
                process; if ``numpy.ndarray`` or ``None``, loads band information
                from ``self.spyfile`` (default: ``None``).
            list_range (``bool``): Whether a band passed as a list is
                interpreted as a range of bands (``True``) or for each
                individual band in the list (``False``). If ``list_range`` is
                ``True``, ``b1``/``wl1`` and ``b2``/``wl2`` should be lists
                with two items, and all bands/wavelegths between the two values
                will be used (default: ``True``).
            print_out (``bool``): Whether to print out the actual bands and
                wavelengths being used in the NDI calculation (default:
                ``True``).

        Returns:
            2-element ``tuple`` containing

            - **array_ratio** (``numpy.ndarray``): Ratio band math array.
            - **metadata** (``dict``): Modified metadata describing the
              ratio array (``array_ratio``).

        Example:
            Load ``hsio`` and ``segment`` modules

            >>> import numpy as np
            >>> import os
            >>> from hs_process import hsio
            >>> from hs_process import segment
            >>> data_dir = r'F:\\nigo0024\Documents\hs_process_demo'
            >>> fname_in = os.path.join(data_dir, 'Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr')
            >>> io = hsio(fname_in)
            >>> my_segment = segment(io.spyfile)

            Calculate a red/near-infrared band ratio using a range of bands
            (i.e., mimicking a broadband sensor) via
            ``segment.band_math_ratio``

            >>> array_ratio, metadata = my_segment.band_math_ratio(wl1=[630, 690], wl2=[800, 860], list_range=True)
            Effective band ratio: (659/830)

            >>> np.nanmean(array_ratio)
            0.10981177

            Notice that 29 spectral bands were consolidated (i.e., averaged) to
            mimic a single broad band. We can take the mean of two bands by
            changing ``list_range`` to ``False``, and this slightly changes the
            result.

            >>> array_ratio, metadata = my_segment.band_math_ratio(wl1=[630, 690], wl2=[800, 860], list_range=False)
            Effective band ratio: (660/830)

            >>> np.nanmean(array_ratio)
            0.113607444

            Show the red/near-infrared ratio image via ``hsio.show_img``

            >>> io.show_img(array_ratio, vmax=0.3)

            .. image:: ../img/segment/ratio_r_nir.png
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
        wl1_list = self._get_wavelength_list(band1_list, list_range=False)
        wl2_list = self._get_wavelength_list(band2_list, list_range=False)
        if print_out is True:
            print('Effective band ratio: ({0:.0f}/{1:.0f})'
                  ''.format(np.mean(wl1_list), np.mean(wl2_list)))

        array_b1 = self.tools.get_spectral_mean(band1_list, array)
        array_b2 = self.tools.get_spectral_mean(band2_list, array)
        with np.errstate(invalid='ignore', divide='ignore'):
            array_ratio = (array_b1/array_b2)
        array_ratio[array_ratio == 0] = np.nan

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

    def composite_band(self, wl1=None, b1=None, spyfile=None, list_range=True,
                       print_out=True):
        '''
        Calculates a composite band from a range of bands or wavelengths.
        Bands/wavelengths can be input as individual bands, a set of bands
        (i.e., list of bands), or a range of bands (i.e., list of two bands
        indicating the lower and upper range).

        Parameters:
            wl1 (``int``, ``float``, or ``list``): the wavelength (or set of
                wavelengths) to consolidate; if ``list``, then all wavelengths
                between two wavelength values are consolidated by calculating
                the mean pixel value across all wavelengths in that range
                (default: ``None``).
            b1 (``int``, ``float``, or ``list``): the band (or set of
                bands) to consolidate; if ``list``, then all bands
                between two band values are consolidated by calculating
                the mean pixel value across all bands in that range (default:
                ``None``).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The datacube to
                process; if ``numpy.ndarray`` or ``None``, loads band information
                from ``self.spyfile`` (default: ``None``).
            list_range (``bool``): Whether a band passed as a list is
                interpreted as a range of bands (``True``) or for each
                individual band in the list (``False``). If ``list_range`` is
                ``True``, ``b1``/``wl1`` and ``b2``/``wl2`` should be lists
                with two items, and all bands/wavelegths between the two values
                will be used (default: ``True``).
            print_out (``bool``): Whether to print out the actual bands and
                wavelengths being used in the NDI calculation (default:
                ``True``).

        Returns:
            2-element ``tuple`` containing

            - **array_b1** (``numpy.ndarray``): Consolidated array.
            - **metadata** (``dict``): Modified metadata describing the
              consolidated array (``array_b1``).
        '''
        wl1 = self._check_bands_wls(wl1, b1, 1)
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
        wl1_list = self._get_wavelength_list(band1_list, list_range=False)
        if print_out is True:
            print('Band 1: {0}'.format(band1_list))
            print('Wavelength 1: {0}'.format(wl1_list))

        array_b1 = self.tools.get_spectral_mean(band1_list, array)
        array_b1[array_b1 == 0] = np.nan

        _, wls_mean = self.tools.get_center_wl(band1_list, spyfile=spyfile,
                                               wls=False)
        metadata['bands'] = 1
        metadata['wavelength'] = [wls_mean]
        metadata['band names'] = [1]
        # self.tools.del_meta_item(metadata, 'band names')
        hist_str = (" -> hs_process.composite_band[<"
                    "label: 'wl1?' value:{0}; "
                    "label: 'list_range?' value:{1}>]"
                    "".format(wl1_list, list_range))
        metadata['history'] += hist_str
        metadata['samples'] = array_b1.shape[1]
        metadata['lines'] = array_b1.shape[0]
        return array_b1, metadata

    def load_spyfile(self, spyfile):
        '''
        Loads a ``SpyFile`` (Spectral Python object) for data access and/or
        manipulation by the ``hstools`` class.

        Parameters:
            spyfile (``SpyFile`` object): The datacube being accessed and/or
                manipulated.

        Example:
            Load ``hsio`` and ``segment`` modules

            >>> from hs_process import hsio
            >>> from hs_process import segment
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)
            >>> my_segment = segment(io.spyfile)

            Load datacube  via ``segment.load_spyfile``

            >>> my_segment.load_spyfile(io.spyfile)
            >>> my_segment.spyfile
            Data Source:   'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip'
        	# Rows:            617
        	# Samples:        1300
        	# Bands:           240
        	Interleave:        BIP
        	Quantization:  32 bits
        	Data format:   float32
        '''
        self.spyfile = spyfile
        self.tools = hstools(spyfile)
        try:
            map_info_set = self.tools.get_meta_set(self.spyfile.metadata['map info'])
            self.spy_ul_e_srs = float(map_info_set[3])
            self.spy_ul_n_srs = float(map_info_set[4])
            self.spy_ps_e = float(map_info_set[5])
            self.spy_ps_n = float(map_info_set[6])
        except KeyError as e:
            print('Map information was not able to be loaded from the '
                  '``SpyFile``. Please be sure the metadata contains the "map '
                  'info" tag with accurate geometric information.\n')
            self.spy_ul_e_srs = None
            self.spy_ul_n_srs = None
            self.spy_ps_e = None
            self.spy_ps_n = None

#    def veg_spectra(self, array_veg, thresh=None, percentile=None,
#                    side='lower', spyfile=None):
#        '''
#        Calculates the average spectra across vegetation pixels
#
#        Parameters:
#            array_veg (``numpy.ndarray``): a single-band image array, presumably
#                that discriminates vegetation pixels from other pixels such as
#                soil, shadow, etc.
#            thresh (``float``): The value for which to base the threshold
#                (default: ``None``).
#            percentile (``float`` or ``int``): The percentile of pixels to mask; if
#                ``percentile``=95 and ``side``='lower', the lowest 95% of pixels
#                will be masked prior to calculating the mean spectra across
#                pixels (default: ``None``; range: 0-100).
#            side (``str``): The side of the threshold or percentile for which to
#                apply the mask. Must be either 'lower' or 'upper'; if 'lower',
#                everything below the threshold/percentile will be masked
#                (default: 'lower').
#        '''
#        if spyfile is None:
#            spyfile = self.spyfile
#        elif isinstance(spyfile, SpyFile.SpyFile):
#            self.load_spyfile(spyfile)
#
##        if remove_shadow is True:
##            shadow_mask, metadata = self.tools.mask_shadow(
##                    shadow_pctl=shadow_pctl, show_histogram=True,
##                    spyfile=spyfile)
##            array_veg = np.ma.array(array_veg, mask=shadow_mask)
##        else:
##            if not isinstance(array_veg, np.ma.core.MaskedArray):
##                array_veg = np.ma.array(array_veg, mask=False)
##            metadata = self.spyfile.metadata
#
#        mask_array, metadata = self.tools.mask_array(
#                array_veg, self.spyfile.metadata, thresh=thresh,
#                percentile=percentile, side=side)
#
#        mask_array_3d = np.empty(spyfile.shape)
#        for band in range(spyfile.nbands):
#            mask_array_3d[:, :, band] = mask_array.mask
#        datacube_masked = np.ma.masked_array(spyfile.load(),
#                                             mask=mask_array_3d)
#        spec_mean = np.nanmean(datacube_masked, axis=(0, 1))
#        spec_std = np.nanstd(datacube_masked, axis=(0, 1))
##        a = spec_mean.reshape(len(spec_mean))
#        spec_mean = pd.Series(spec_mean)
#        spec_std = pd.Series(spec_std)
#
#        return spec_mean, spec_std, datacube_masked, metadata

#    def mask_datacube(self, mask, spyfile=None):
#        '''
#        DO NOT USE; USE hstools.mask_datacube() INSTEAD AND PASS A MASK.
#
#        Applies ``mask`` to ``spyfile``, then returns the datcube (as a np.array)
#        and the mean spectra
#
#        Parameters:
#            mask (``numpy.ndarray``): the mask to apply to ``spyfile``; if ``mask``
#                does not have similar dimensions to ``spyfile``, the first band
#                (i.e., first two dimensions) of ``mask`` will be repeated n times
#                to match the number of bands of ``spyfile``.
#            spyfile (``SpyFile`` object): The datacube being accessed and/or
#                manipulated.
#        '''
#        if spyfile is None:
#            spyfile = self.spyfile
#        elif isinstance(spyfile, SpyFile.SpyFile):
#            self.load_spyfile(spyfile)
#
#        if isinstance(mask, np.ma.masked_array):
#            mask = mask.mask
#        if mask.shape != spyfile.shape:
#            mask_1d = mask.copy()
#            mask = np.empty(spyfile.shape)
#            for band in range(spyfile.nbands):
#                mask[:, :, band] = mask_1d
#
#        datacube_masked = np.ma.masked_array(spyfile.load(), mask=mask)
#        spec_mean = np.nanmean(datacube_masked, axis=(0, 1))
#        spec_std = np.nanstd(datacube_masked, axis=(0, 1))
#        spec_mean = pd.Series(spec_mean)
#        spec_std = pd.Series(spec_std)
#        return spec_mean, spec_std, datacube_masked

# if __name__ == '__main__':
#     import doctest
#     doctest.testmod()