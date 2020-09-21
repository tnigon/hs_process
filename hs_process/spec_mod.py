# -*- coding: utf-8 -*-
from copy import deepcopy
from math import factorial
import itertools
import numpy as np
import os
import pandas as pd
from scipy import interpolate
import spectral.io.envi as envi
import spectral.io.spyfile as SpyFile

from hs_process.utilities import hstools


class spec_mod(object):
    '''
    Class for manipulating data within the spectral domain, which
    is usually pixel-based.
    '''
    def __init__(self, spyfile):
        self.spyfile = spyfile
        self.tools = hstools(spyfile)

    def _metadata_clip(self, wl_bands, meta_bands):
        '''Modifies metadata for spectral_clip() function.'''
        metadata = deepcopy(self.tools.spyfile.metadata)
        hist_str = (" -> hs_process.spectral_clip[<SpecPyFloatText label: "
                    "'wl_bands?' value:{0}>]".format(wl_bands))
        metadata['history'] += hist_str
        metadata['bands'] = len(meta_bands)

        band = []
        wavelength = []
        for idx, (key, val) in enumerate(meta_bands.items()):
            band.append(idx + 1)
            wavelength.append(val)
        band_str = '{' + ', '.join(str(b) for b in band) + '}'
        wavelength.sort()
        wavelength_str = '{' + ', '.join(str(wl) for wl in wavelength) + '}'
        metadata['band names'] = band_str
        metadata['wavelength'] = wavelength_str
        # self.tools.spyfile.metadata = metadata
        return metadata

    def _metadata_spectral_mimic(self, sensor, meta_bands_sensor):
        '''Modifies metadata for spectral_bin() function.'''
        metadata = deepcopy(self.tools.spyfile.metadata.copy())
        band_names = list(meta_bands_sensor.keys())
        hist_str = (" -> hs_process.spectral_mimic[<"
                    "SpecPyFloatText label: 'sensor?' value:{0}; "
                    "SpecPyFloatText label: 'band_names?' value:{1}>]"
                    "".format(sensor, band_names))
        metadata['history'] += hist_str
        metadata['bands'] = len(band_names)
        wavelength = list(meta_bands_sensor.values())
        band_str = '{' + ', '.join(str(b) for b in band_names) + '}'
        wavelength_str = '{' + ', '.join(str(wl) for wl in wavelength) + '}'
        metadata['band names'] = band_str
        metadata['wavelength'] = wavelength_str
        # self.tools.spyfile.metadata = metadata
        return metadata

    def _metadata_smooth(self, window_size, order):
        '''Modifies metadata for spectral_smooth() function.'''
        metadata = deepcopy(self.tools.spyfile.metadata.copy())
        hist_str = (" -> hs_process.spectral_smooth[<"
                    "SpecPyFloatText label: 'window_size?' value:{0}; "
                    "SpecPyFloatText label: 'polynomial_order?' value:{1}>"
                    "]".format(window_size, order))
        metadata['history'] += hist_str
        metadata['bands'] = self.spyfile.nbands
        # self.tools.spyfile.metadata = metadata
        return metadata

    def _mimic_center_wl(self, center_wl, wl_hs, spec_response):
        '''Finds the center wavelength based on the method of ``center_wl``.'''
        if center_wl == 'weighted':
            wl = np.average(wl_hs, weights=spec_response)
        elif center_wl == 'peak':
            wl = wl_hs[np.argmax(spec_response)]
        return wl

    def _mimic_get_response(self, sensor, df_band_response, col_wl):
        '''Gets ``df_band_response`` based on the ``sensor`` passed.'''
        if sensor == 'custom':
            msg = ('``col_wl`` ({0}) must be a column in ``df_band_response``.'
                   ''.format(col_wl))
            assert col_wl in df_band_response.columns, msg
        else:
            dir_data = self.tools.dir_data()
            name_response = sensor + '_band_response.csv'
            fname_response = os.path.join(dir_data, name_response)
            df_band_response = pd.read_csv(fname_response)
            col_wl = 'wl_nm'

        df = df_band_response.rename(columns = {col_wl:'wl_nm'})

        wl_min = min(self.tools.meta_bands.values())-100
        wl_max = max(self.tools.meta_bands.values())+100  # adding because interp needs some room
        df = df[df['wl_nm'] >= wl_min]
        df = df[df['wl_nm'] <= wl_max]
        s = (df==0).all(0)  # Checks if columns are all zero or not
        df = df.drop(columns=s[s].index)  # Removes all zero cols

        band_names = list(df.columns)
        band_names.remove('wl_nm')
        return df, band_names

    def _savitzky_golay(self, y, window_size=5, order=2, deriv=0, rate=1):
        '''
        Smooth (and optionally differentiate) data with a Savitzky-Golay
        filter. The Savitzky-Golay filter removes high frequency noise from
        data. It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.

        Parameters:
        y (``numpy.array``; shape (N,)): the values of the time history of the
            signal.
        window_size (``int``): the length of the window; must be an odd integer
            number (default: 5).
        order (``int``): the order of the polynomial used in the filtering; must
            be less than ``window_size`` - 1 (default: 2).
        deriv (``int``): the order of the derivative to compute (default: 0,
              means only smoothing).

        Returns:
        ys (``ndarray``; shape (N)): the smoothed signal (or it's n-th
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
            array = self.spyfile.open_memmap()
        array_2d = array.reshape(array.shape[0]*array.shape[1], array.shape[2])
        array_2d_temp = array_2d.copy()
        for idx, row in enumerate(array_2d):
            array_2d_temp[idx, :] = self._savitzky_golay(
                    row, window_size=window_size, order=order)
#            sns.lineplot(list(hs.meta_bands.keys()), array_2d[1000])
#            sns.lineplot(list(hs.meta_bands.keys()), array_2d_temp[1000])
        return array_2d_temp.reshape((array.shape))

    def load_spyfile(self, spyfile):
        '''
        Loads a ``SpyFile`` (Spectral Python object) for data access and/or
        manipulation by the ``hstools`` class.

        Parameters:
            spyfile (``SpyFile`` object): The datacube being accessed and/or
                manipulated.

        Example:
            Load and initialize the ``hsio`` and ``spec_mod`` modules

            >>> from hs_process import hsio
            >>> from hs_process import spec_mod
            >>> fname_in = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio(fname_in)
            >>> my_spec_mod = spec_mod(io.spyfile)

            Load datacube

            >>> my_spec_mod.load_spyfile(io.spyfile)
            >>> my_spec_mod.spyfile
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
            self.spy_ul_e_srs = float(self.spyfile.metadata['map info'][3])
            self.spy_ul_n_srs = float(self.spyfile.metadata['map info'][4])
            self.spy_ps_e = float(self.spyfile.metadata['map info'][5])
            self.spy_ps_n = float(self.spyfile.metadata['map info'][6])
        except KeyError as e:
            print('Map information was not able to be loaded from the '
                  '``SpyFile``. Please be sure the metadata contains the "map '
                  'info" tag with accurate geometric information.\n')
            self.spy_ul_e_srs = None
            self.spy_ul_n_srs = None
            self.spy_ps_e = None
            self.spy_ps_n = None

    def spectral_clip(self, wl_bands=[[0, 420], [760, 776], [813, 827],
                                      [880, 1000]], spyfile=None):
        '''
        Removes/clips designated wavelength bands from the hyperspectral
        datacube.

        Parameters:
            wl_bands (``list`` or ``list`` of ``lists``): minimum and maximum
                wavelenths to clip from image; if multiple groups of
                wavelengths should be cut, this should be a list of lists. For
                example, wl_bands=[760, 776] will clip all bands greater than
                760.0 nm and less than 776.0 nm;
                wl_bands = [[0, 420], [760, 776], [813, 827], [880, 1000]]
                will clip all band less than 420.0 nm, bands greater than 760.0
                nm and less than 776.0 nm, bands greater than 813.0 nm and less
                than 827.0 nm, and bands greater than 880 nm (default).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The data cube to
                clip; if ``numpy.ndarray`` or ``None``, loads band information
                from ``spec_mod.spyfile`` (default: ``None``).

        Returns:
            2-element ``tuple`` containing

            - **array_clip** (``numpy.ndarray``): Clipped datacube.
            - **metadata** (``dict``): Modified metadata describing the clipped
              hyperspectral datacube (``array_clip``).

        Example:
            Load and initialize ``hsio`` and ``spec_mod``

            >>> from hs_process import hsio
            >>> from hs_process import spec_mod
            >>> fname_hdr = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io1 = hsio()
            >>> io1.read_cube(fname_hdr)
            >>> my_spec_mod = spec_mod(io1.spyfile)

            Using ``spec_mod.spectral_clip``, clip all spectral bands below
            *420 nm* and above *880 nm*, as well as the bands near the oxygen
            absorption (i.e., *760-776 nm*) and water absorption
            (i.e., *813-827 nm*) regions.

            >>> array_clip, metadata = my_spec_mod.spectral_clip(
                    wl_bands=[[0, 420], [760, 776], [813, 827], [880, 1000]])

            Save the clipped datacube

            >>> fname_hdr_clip = r'F:\\nigo0024\Documents\hs_process_demo\spec_mod\Wells_rep2_20180628_16h56m_pika_gige_7-clip.bip.hdr'
            >>> io1.write_cube(fname_hdr_clip, array_clip, metadata)
            Saving F:\nigo0024\Documents\hs_process_demo\spec_mod\Wells_rep2_20180628_16h56m_pika_gige_7-clip.bip

            Initialize a second instance of ``hsio`` and read in the clipped
            cube to compare the clipped cube to the unclipped cube

            >>> io2 = hsio()  # initialize a second instance to compare cubes
            >>> io2.read_cube(fname_hdr_clip)
            >>> io1.spyfile
            Data Source:   'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip'
        	# Rows:            617
        	# Samples:        1300
        	# Bands:           240
        	Interleave:        BIP
        	Quantization:  32 bits
        	Data format:   float32

            The unclipped cube (above) has 240 spectral bands, while the
            clipped cube (below) has 210.

            >>> io2.spyfile
            Data Source:   'F:\\nigo0024\Documents\hs_process_demo\spec_mod\Wells_rep2_20180628_16h56m_pika_gige_7-clip.bip'
        	# Rows:            617
        	# Samples:        1300
        	# Bands:           210
        	Interleave:        BIP
        	Quantization:  32 bits
        	Data format:   float32
        '''
        if spyfile is None:
            spyfile = self.spyfile
            array = self.spyfile.open_memmap()
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = self.spyfile.open_memmap()
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()
            spyfile = self.spyfile

        if isinstance(wl_bands[0], list):
            spec_clip_groups = [self.tools.get_band_range(
                    grp, index=True) for grp in wl_bands]
            spec_clip = list(itertools.chain(*spec_clip_groups))
        else:
            spec_clip = self.tools.get_band_range(wl_bands, index=True)

        meta_bands = self.tools.meta_bands.copy()
        for k in self.tools.get_band_num(spec_clip):
            meta_bands.pop(k, None)
#        tools.meta_bands = meta_bands
        array_clip = np.delete(array, spec_clip, axis=2)
        metadata = self._metadata_clip(wl_bands, meta_bands)
        return array_clip, metadata

    def spectral_mimic(self, sensor='sentera_6x', df_band_response=None,
                       col_wl='wl_nm', center_wl='peak', spyfile=None):
        '''
        Mimics the response of a multispectral sensor based on transmissivity
        of sensor bands across a range of wavelength values by calculating its
        weighted average response and interpolating the hyperspectral response.

        Parameters:
            sensor (``str``): Should be one of
                ["sentera_6x", "micasense_rededge_3", "sentinel-2a",
                "sentinel-2b", "custom"]; if "custom", ``df_band_response``
                and ``col_wl`` must be passed.
            df_band_response (``pd.DataFrame``): A DataFrame that contains the
                transmissivity (%) for each sensor band (as columns) mapped to
                the continuous wavelength values (as rows). Required if
                ``sensor`` is  "custom", ignored otherwise.
            col_wl (``str``): The column of ``df_band_response`` denoting the
                wavlengths (default: 'wl_nm').
            center_wl (``str``): Indicates how the center wavelength of each
                band is determined. If ``center_wl`` is "peak", the point at
                which transmissivity is at its maximum is used as the center
                wavelength. If ``center_wl`` is "weighted", the weighted
                average is used to compute the center wavelength. Must be one
                of ["peak", "weighted"] (``default: "peak"``).
            spyfile (``SpyFile`` object): The datacube being accessed and/or
                manipulated.

        Returns:
            2-element ``tuple`` containing

            - **array_multi** (``numpy.ndarray``): Mimicked datacube.
            - **metadata** (``dict``): Modified metadata describing the
              mimicked spectral array (``array_multi``).

        Example:
            Load and initialize ``hsio`` and ``spec_mod``

            >>> from hs_process import hsio
            >>> from hs_process import spec_mod
            >>> fname_hdr = r'F:\nigo0024\Documents\GitHub\hs_process\hs_process\data\Wells_rep2_20180628_16h56m_test_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> fname_hdr = r'F:\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio()
            >>> io.read_cube(fname_hdr)
            >>> my_spec_mod = spec_mod(io.spyfile)

            Use spec_mod.spectral_mimic to mimic the Sentinel-2A spectral
            response function.

            >>> array_s2a, metadata_s2a = my_spec_mod.spectral_mimic(sensor='sentinel-2a', center_wl='weighted')

            Plot the mean spectral response of the hyperspectral image to that
            of the mimicked Sentinel-2A image bands (mean calculated across the
            entire image).

            >>> import seaborn as sns
            >>> from ast import literal_eval
            >>> array_hs = my_spec_mod.spyfile.open_memmap()
            >>> mean_hs = array_hs.mean(axis=(0,1))*100
            >>> mean_s2a = array_s2a.mean(axis=(0,1))*100
            >>> x1 = my_spec_mod.tools.get_wavelength_range([0,239])  # list of wavelength values
            >>> x2 = sorted(list(literal_eval(metadata_s2a['wavelength'])))  # list of Sentinel-2A bands
            >>> ax = sns.lineplot(x=x1, y=mean_hs, label='Hyperspectral')
            >>> ax = sns.lineplot(x=x2, y=mean_s2a, ax=ax, label='Sentinel-2A', marker='o', ms=6)
            >>> _ = ax.set(xlabel='Wavelength (nm)', ylabel='Reflectance (%)')

            .. image:: ../img/spec_mod/spectral_mimic_hs_vs_sentinel-2a.png
        '''
        if spyfile is None:
            spyfile = self.spyfile
            array = self.spyfile.open_memmap()
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = self.spyfile.open_memmap()
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()
            spyfile = self.spyfile

        msg1 = ('``sensor`` ({0}) must be one of ["sentera_6x", "custom"].'
                ''.format(sensor))
        msg2 = ('``center_wl`` ({0}) must be one of ["peak", "weighted"].'
                ''.format(center_wl))
        assert sensor in ['sentera_6x', 'micasense_rededge_3', 'sentinel-2a',
                          'sentinel-2b', 'custom'], msg1
        assert center_wl in ['peak', 'weighted'], msg2

        df, band_names = self._mimic_get_response(sensor, df_band_response,
                                                  col_wl)
        # Normalizes cells along columns so column sum equals 1.0
        df_norm = df.loc[:,:].div(df.sum(axis=0).drop('wl_nm'))
        df_norm['wl_nm'] = df['wl_nm']


        # df_temp = df_norm.cumsum(axis=0)
        # df_temp['wl_nm'] = df['wl_nm']

        array_multi = np.empty((self.spyfile.nrows,
                                self.spyfile.ncols,
                                len(band_names)))
        # interpolate and sum each pixel to get mimicked value
        meta_bands = self.tools.meta_bands.copy()
        meta_bands_sensor = {}
        wl_sensor = df_norm['wl_nm'].values
        wl_hs = list(meta_bands.values())
        bands_remove = {}
        for idx, band_name in enumerate(band_names):
            y_sensor = df_norm[band_name].values
            f = interpolate.interp1d(wl_sensor, y_sensor)
            spec_response = f(wl_hs)  # this is the response that can be multiplied by every pixel
            if np.count_nonzero(spec_response) > 0:
                array_multi[:,:,idx] = np.average(array, weights=spec_response, axis=2)
                meta_bands_sensor[band_name] = self._mimic_center_wl(center_wl, wl_hs, spec_response)
            else:
                bands_remove[idx] = band_name
        for idx, band_name in bands_remove.items():
            array_multi = np.delete(array_multi, idx, axis=2)
        metadata = self._metadata_spectral_mimic(sensor, meta_bands_sensor)
        return array_multi, metadata

    def spectral_smooth(self, window_size=11, order=2, spyfile=None):
        '''
        Performs Savitzky-Golay smoothing on the spectral domain.

        Parameters:
            window_size (``int``): the length of the window; must be an odd
                integer number (default: 11).
            order (``int``): the order of the polynomial used in the filtering;
                must be less than ``window_size`` - 1 (default: 2).
            spyfile (``SpyFile`` object or ``numpy.ndarray``): The data cube to
                clip; if ``numpy.ndarray`` or ``None``, loads band information
                from ``spec_mod.spyfile`` (default: ``None``).

        Returns:
            2-element ``tuple`` containing

            - **array_smooth** (``numpy.ndarray``): Clipped datacube.
            - **metadata** (``dict``): Modified metadata describing the smoothed
              hyperspectral datacube (``array_smooth``).

        Note:
            Because the smoothing operation is performed for every pixel
            individually, this function may take several minutes for large
            images.

        Example:
            Load and initialize ``hsio`` and ``spec_mod``

            >>> from hs_process import hsio
            >>> from hs_process import spec_mod
            >>> fname_hdr = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
            >>> io = hsio()
            >>> io.read_cube(fname_hdr)
            >>> my_spec_mod = spec_mod(io.spyfile)

            Use ``spec_mod.spectral_smooth`` to perform a *Savitzky-Golay*
            smoothing operation across the hyperspectral spectral signature.

            >>> array_smooth, metadata = my_spec_mod.spectral_smooth(
                    window_size=11, order=2)

            Save the smoothed datacube using ``hsio.write_cube``

            >>> fname_hdr_smooth = r'F:\\nigo0024\Documents\hs_process_demo\spec_mod\Wells_rep2_20180628_16h56m_pika_gige_7-smooth.bip.hdr'
            >>> io.write_cube(fname_hdr_smooth, array_smooth, metadata)
            Saving F:\\nigo0024\Documents\hs_process_demo\spec_mod\Wells_rep2_20180628_16h56m_pika_gige_7-smooth.bip

            Open smoothed datacube in Spectronon software to visualize the
            result of the smoothing for a specific pixel.

            .. image:: ../img/spec_mod/spectral_smooth_before.png

            Before smoothing (the spectral curve of the pixel at the *800th
            column/sample* and *200th row/line* is plotted)

            .. image:: ../img/spec_mod/spectral_smooth_after.png

            And after smoothing (the spectral curve of the pixel at the *800th
            column/sample* and *200th row/line* is plotted)
        '''
        if spyfile is None:
            spyfile = self.spyfile
            array = self.spyfile.open_memmap()
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = self.spyfile.open_memmap()
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()
            spyfile = self.spyfile

        array_smooth = self._smooth_image(array, window_size, order)
        metadata = self._metadata_smooth(window_size, order)
        return array_smooth, metadata


# TODO: Add normalization function for light scattering
# https://link.springer.com/chapter/10.1007/978-1-4939-2836-1_4#Sec12