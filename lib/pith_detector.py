"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
from pathlib import Path

from lib.fft_local_orientation import LocalOrientationEstimation, local_orientation_estimation
from lib.accumulator_space import AccumulationSpace
from lib.peak_estimator import find_peak




class PithDetector:
    def __init__(self, img_in=None, mask=None, block_overlap=0.35, block_width_size=15, block_height_size=15,
                 fft_peak_th=0.6, lo_method=LocalOrientationEstimation.pca, lo_certainty_th=0.5, acc_type=0,
                 peak_blur_sigma=5, debug=False, output_dir=None):
        """
        Implementation of the pith detection algorithm described in Reference Paper. Algorithm 1 in paper
        :param img_in: input image
        :param mask: input background mask
        :param block_overlap: overlap between patches
        :param block_width_size: Patches (blocks) width size
        :param block_height_size: Patches (blocks) height size
        :param fft_peak_th: FFT frequency filter threshold
        :param lo_method: method to use for local orientation estimation
        :param lo_certainty_th: threshold for local orientation estimation
        :param acc_type: type of accumulator to use
        :param peak_blur_sigma: sigma for peak blurring. Guassian kernel size
        :param output_dir: output directory
        :param debug: debug flag
        """
        self.img_in = img_in
        self.mask = mask
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        self.fft_peak_th = fft_peak_th
        self.block_width_size = block_width_size
        self.block_height_size = block_height_size
        self.block_overlap = block_overlap
        self.debug = debug
        self.lo_method = lo_method
        self.lo_certainty_th = lo_certainty_th
        self.acc_type = acc_type
        self.peak_blur_sigma = peak_blur_sigma

    def local_orientation(self, img_in, mask, block_overlap, block_width_size, block_height_size, fft_peak_th,
                          lo_method, lo_certainty_th, debug=True):
        """
        Create debug local orientation directory and run local orientation estimation algorithm (Algorithm 2)
        :param img_in: rgb input image
        :param mask: background mask image
        :param block_overlap: overlapping between blocks
        :param block_width_size: block width size (px)
        :param block_height_size: block height size (px)
        :param fft_peak_th: FFT frequency filter threshold
        :param lo_method: local orientation method (pca, peak, lsq, wlsq)
        :param lo_certainty_th: local orientation certainty threshold
        :param debug: debug flag
        :return: List of lines of local orientation
        """
        # 0.0 Create debug directory
        lo_dir = Path(self.output_dir) / 'local_orientation'
        lo_dir.mkdir(exist_ok=True, parents=True)

        # 1.0 Compute local orientation. Algorithm 2 in Paper.
        l_lo = local_orientation_estimation(img_in = img_in, mask = mask, block_overlap = block_overlap,
                                     block_width_size = block_width_size,
                                     block_height_size = block_height_size, lo_method = lo_method,
                                     lo_certainty_th = lo_certainty_th, fft_peak_th = fft_peak_th,
                                     output_dir = str(lo_dir), debug = debug)

        return l_lo

    def accumulation_space(self, l_lo, acc_type, debug=True):
        # 3.0 Compute accumulation space. Section 2.3 in Paper. Algorithm 3 in Paper
        as_dir = Path(self.output_dir) / 'accumulation_space'
        if debug:
            as_dir.mkdir(exist_ok=True, parents=True)
        as_obj = AccumulationSpace(l_lo=l_lo, output_dir=str(as_dir), img = self.img_in, mask=self.mask, type=acc_type, debug=debug)

        return as_obj.run()

    def peak_estimation(self, m_accumulation_space, peak_blur_sigma, debug=True):
        # 3.0 compute accumulation space. Section 2.3 in Paper

        peak_location = find_peak( m_accumulation_space, self.img_in, str(self.output_dir), sigma = peak_blur_sigma)

        return peak_location


    def run(self):
        """Implementations of Block Area Selection - BAS algorithm described in Reference Paper.
        Algorithm 1 in Paper"""
        # Line 1
        l_lo = self.local_orientation(self.img_in, self.mask, self.block_overlap, self.block_width_size,
                        self.block_height_size, self.fft_peak_th,  self.lo_method,  self.lo_certainty_th,
                        debug=self.debug)

        # Line 2
        m_accumulation_space = self.accumulation_space(l_lo, self.acc_type, debug=self.debug)

        # Line 3
        peak = self.peak_estimation(m_accumulation_space, self.peak_blur_sigma)

        return peak