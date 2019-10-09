# -*- coding: utf-8 -*-
from math import factorial
import itertools
import numpy as np
import os
import spectral.io.envi as envi


class Spec_mod(object):
    '''
    Class for manipulating data within the spectral domain (usually
    pixel-based)
    '''
    def __init__(self, img_sp):
        self.img_sp = img_sp

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
            print('\n\n{0}'.format(interleave))
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
                                 spectra_smooth=True, window_size=19, order=2,
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
        print(fname_out_envi)
        print('Smoothing and spectrally clipping image: {0}\n'
              ''.format(name_print))

        self.spec_clip = spec_clip
        meta_bands = self.meta_bands.copy()
        for k in self._get_band_num(spec_clip):
            meta_bands.pop(k, None)
        self.meta_bands = meta_bands
        array_clip = np.delete(self.img_sp.asarray(), spec_clip, axis=2)
        if spectra_smooth is True:
            self.array_smooth = self._smooth_image(array_clip, window_size=19,
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