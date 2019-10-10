# -*- coding: utf-8 -*-
import numpy as np
import os


class segment(object):
    '''
    Class for aiding in the segmentation/masking of image data to include
    pixels that are of most interest.
    '''
    def __init__(self, img_sp):
        self.img_sp = img_sp

    def _mask_array(self, array, thresh=0.55, side='lower'):
        '''
        Creates a masked numpy array based on a threshold value
        '''
        if side == 'lower':
            mask_array = np.ma.array(array, mask=array <= 0.55)
        elif side == 'upper':
            mask_array = np.ma.array(array, mask=array > 0.55)
        unmasked_pct = 100 * (mask_array.count() /
                              (array.shape[0]*array.shape[1]))
        print('Proportion unmasked pixels: {0:.2f}%'.format(unmasked_pct))
        return mask_array

    def band_math_ratio(self, b1, b2, base_dir_out=None, name=None,
                        save_out=True, interleave='bip'):
        '''
        Calculates a simple ratio spectral index from two input band
        wavelengths

        Parameters:
            b1 (`int` or `float`): the first band to be used in the index; this
                should be the numerator (required).
            b2 (`int` or `float`): the second band to be used in the index;
                this should be the denominator (required).
            base_dir_out (`str`): directory path to save file to; if `None`,
                a new folder ("ratio") will be created in the directory of
                the original image file (default: `None`).
            name (`str`): text to append to end of filename; used to describe
                this particular manipulation (default: 'ndi_b1_b2' where b1 and
                b2 denote band wavelengths used in index).
            save_out (`bool): indicates whether manipulated image should be
                saved to disk (defaul: True).
            interleave (`str`): interleave (and file extension) of manipulated
                file; (default: 'bip').
        '''
        base_dir_out, name_print = self._save_file_setup(
                base_dir_out, folder_name='band_math')
        band1, band2, wl1, wl2 = self._get_band_info_consolidate(b1, b2)
        if name is None:
            name = 'ratio_{0:.0f}_{1:.0f}'.format(wl1, wl2)
        fname_out_envi = os.path.join(
            base_dir_out, (name_print + '_' + str(name) + '.' + interleave))
        print('Calculating normalized difference index for {0}: '
              '{1:.0f}/{2:.0f}'.format(name_print, wl1, wl2))

        if self.array_smooth is not None:
            array = self.array_smooth
        else:
            array = self.img_sp.asarray()
        array_b1 = self._get_band_mean(array, band1)
        array_b2 = self._get_band_mean(array, band2)
        array_index = (array_b1/array_b2)

        geotransform_out = self.geotransform
        if save_out is True:
            self._write_envi(array_index, fname_out_envi, geotransform_out,
                             name, interleave=interleave, modify_hdr=False)
        return array_index

    def band_math_ndi(self, b1=780, b2=559, b3=None, b4=None, b5=None,
                      base_dir_out=None, name=None,
                      save_out=True, interleave='bip'):
        '''
        Calculates the spectral index from a list of bands and the "form" of
        the index

        Parameters:
            b1 (`int` or `float`): the first band to be used in the index
                (required).
            b2 (`int` or `float`): the second band to be used in the index
                (required).
            base_dir_out (`str`): directory path to save file to; if `None`,
                a new folder ("band_math") will be created in the directory of
                the original image file (default: `None`).
            name (`str`): text to append to end of filename; used to describe
                this particular manipulation (default: 'ndi_b1_b2' where b1 and
                b2 denote band wavelengths used in index).
            save_out (`bool): indicates whether manipulated image should be
                saved to disk (defaul: True).
            interleave (`str`): interleave (and file extension) of manipulated
                file; (default: 'bip').
        '''
        base_dir_out, name_print = self._save_file_setup(
                base_dir_out, folder_name='band_math')
        band1, band2, wl1, wl2 = self._get_band_info_consolidate(b1, b2)
        if name is None:
            name = 'ndi_{0:.0f}_{1:.0f}'.format(wl1, wl2)
        fname_out_envi = os.path.join(
            base_dir_out, (name_print + '_' + str(name) + '.' + interleave))
        print('Calculating normalized difference index for {0}: '
              '({1:.0f}-{2:.0f})/({1:.0f}+{2:.0f})'.format(name_print,
                                                           wl1, wl2))
        if self.array_smooth is not None:
            array = self.array_smooth
        else:
            array = self.img_sp.asarray()
        array_b1 = self._get_band_mean(array, band1)
        array_b2 = self._get_band_mean(array, band2)
        array_index = (array_b1-array_b2)/(array_b1+array_b2)

        geotransform_out = self.geotransform
        if save_out is True:
            self._write_envi(array_index, fname_out_envi, geotransform_out,
                             name, interleave=interleave, modify_hdr=False)
        return array_index

    def veg_spectra(self, array_gndvi, thresh=0.55, side='lower'):
        '''
        Gets average spectra across vegetation pixels
        '''
        mask_array = self._mask_array(array_gndvi, thresh=thresh, side=side)

        self.mask_array_3d = np.empty(self.array_smooth.shape)
        for band in range(self.img_sp.nbands):
            self.mask_array_3d[:, :, band] = mask_array.mask
        array_smooth_masked = np.ma.masked_array(self.array_smooth,
                                                 mask=self.mask_array_3d)
        veg_spectra = np.mean(array_smooth_masked, axis=(0, 1))
        return veg_spectra
