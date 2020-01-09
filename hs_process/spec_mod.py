# -*- coding: utf-8 -*-
from math import factorial
import itertools
import numpy as np
import os
import spectral.io.envi as envi
import spectral.io.spyfile as SpyFile

from hs_process.utilities import hstools


class spec_mod(object):
    '''
    Class for manipulating data within the spectral domain (usually
    pixel-based)
    '''
    def __init__(self, spyfile):
        self.spyfile = spyfile
        self.tools = hstools(spyfile)

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

    def spectral_clip(self, wl_bands=[[0, 420], [760, 776], [813, 827],
                                      [880, 1000]], spyfile=None):
        '''
        Removes/clips designated wavelength bands from the hyperspectral
        datacube.

        Parameters:
            wl_bands (`list` or `list of lists`): minimum and maximum
                wavelenths to clip from image; if multiple groups of
                wavelengths should be cut, this should be a list of lists. For
                example, wl_bands=[760, 776] will clip all bands greater than
                760.0 nm and less than 776.0 nm;
                wl_bands = [[0, 420], [760, 776], [813, 827], [880, 1000]]
                will clip all band less than 420.0 nm, bands greater than 760.0
                nm and less than 776.0 nm, bands greater than 813.0 nm and less
                than 827.0 nm, and bands greater than 880 nm (default).
            spyfile (`SpyFile` object or `numpy.ndarray`): The data cube to
                clip; if `numpy.ndarray` or `None`, loads band information from
                `self.spyfile` (default: `None`).

        Returns:
            2-element `tuple` containing

            - **array_clip** (`numpy.ndarray`): Clipped datacube.
            - **metadata** (`dict`): Modified metadata describing the clipped
              hyperspectral datacube (`array_clip`).
        '''
        if spyfile is None:
            spyfile = self.spyfile
            array = self.spyfile.load()
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = self.spyfile.load()
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

        metadata = self.tools.spyfile.metadata
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
        self.tools.spyfile.metadata = metadata

        return array_clip, metadata

    def spectral_smooth(self, window_size=11, order=2, spyfile=None):
        '''
        Performs Savitzky-Golay smoothing on the spectral domain

        Parameters:
            window_size (`int`): the length of the window; must be an odd integer
                number (default: 11).
            order (`int`): the order of the polynomial used in the filtering; must
                be less than `window_size` - 1 (default: 2).
            spyfile (`SpyFile` object or `numpy.ndarray`): The data cube to
                clip; if `numpy.ndarray` or `None`, loads band information from
                `self.spyfile` (default: `None`).

        Returns:
            2-element `tuple` containing

            - **array_smooth** (`numpy.ndarray`): Clipped datacube.
            - **metadata** (`dict`): Modified metadata describing the smoothed
              hyperspectral datacube (`array_smooth`).
        '''
        if spyfile is None:
            spyfile = self.spyfile
            array = self.spyfile.load()
        elif isinstance(spyfile, SpyFile.SpyFile):
            self.load_spyfile(spyfile)
            array = self.spyfile.load()
        elif isinstance(spyfile, np.ndarray):
            array = spyfile.copy()
            spyfile = self.spyfile

        array_smooth = self._smooth_image(array, window_size, order)

        metadata = self.tools.spyfile.metadata
        hist_str = (" -> hs_process.spectral_smooth[<"
                    "SpecPyFloatText label: 'window_size?' value:{0}; "
                    "SpecPyFloatText label: 'polynomial_order?' value:{1}>"
                    "]".format(window_size, order))
        metadata['history'] += hist_str
        metadata['bands'] = self.spyfile.nbands
        self.tools.spyfile.metadata = metadata

        return array_smooth, metadata

# TODO: Add normalization function for light scattering
# https://link.springer.com/chapter/10.1007/978-1-4939-2836-1_4#Sec12