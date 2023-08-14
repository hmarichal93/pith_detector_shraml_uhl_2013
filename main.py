"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

Implementation of the paper: "Pith Estimation on Rough Log End images using Local Fourier Spectrum Analysis" by Rudolf Schraml and Andreas Uhl.
http://dx.doi.org/10.2316/P.2013.797-012
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import time

from lib.image import resize_image_using_pil_lib, Color
from lib.fft_local_orientation import LocalOrientationEstimation
from lib.pith_detector import PithDetector



def main(filename, output_dir, new_shape=640, fft_peak_th=0.8, block_width_size=10,
         block_height_size=10, block_overlap=0.25, lo_method=LocalOrientationEstimation.pca,
         lo_certainty_th=0.7, peak_blur_sigma=5, acc_type=0, debug=True):

    to = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # 1.0 load image
    img_in = cv2.imread(filename)
    o_height, o_width = img_in.shape[:2]
    # 1.1 resize image
    img_in = resize_image_using_pil_lib(img_in, height_output=new_shape, width_output=new_shape)
    cv2.imwrite(str(output_dir / 'resized.png'), img_in)

    # 2.0 segment image. Input image is RGB with white background colored in white
    mask = np.where( img_in == Color.gray_white, 0, Color.gray_white).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.uint8)

    # 3.0 Pith detector. Implementation of the paper:
    # "Pith Estimation on Rough Log End images using Local Fourier Spectrum Analysis" by Rudolf Schraml and Andreas Uhl.
    pith_detector = PithDetector(img_in=img_in, mask=mask, block_overlap=block_overlap,
                                 block_width_size=block_width_size, block_height_size=block_height_size,
                                 fft_peak_th=fft_peak_th, lo_method=lo_method, lo_certainty_th=lo_certainty_th,
                                 acc_type=acc_type, peak_blur_sigma=peak_blur_sigma, debug=debug,
                                 output_dir=str(output_dir))
    peak = pith_detector.run()

    tf = time.time()
    # 4.0 save results
    convert_original_scale = lambda peak: (np.array(peak) * np.array([o_width/new_shape,o_height/new_shape])).tolist()
    data = {'coarse': convert_original_scale(peak), 'exec_time(s)':tf-to}
    df = pd.DataFrame(data)
    df.to_csv(str(output_dir / 'pith.csv'), index=False)

    return peak


import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pith detector')
    parser.add_argument('--filename', type=str, required=True, help='image filename')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')

    #method parameters
    parser.add_argument('--new_shape', type=int, default=1000, help='new shape')
    parser.add_argument('--fft_peak_th', type=float, default=0.8, help='fourier transform peak threshold')
    parser.add_argument('--block_width_size', type=int, default=100, help='width partition size (px)')
    parser.add_argument('--block_height_size', type=int, default=100, help='height partition size (px)')
    parser.add_argument('--block_overlap', type=float, default=0.1, help='block overlapping')
    parser.add_argument('--lo_method', type=str, default='pca', help='lo method')
    parser.add_argument('--lo_certainty_th', type=float, default=0.9, help='lo certainty threshold')
    parser.add_argument('--peak_blur_sigma', type=int, default=3, help='peak blur sigma')
    parser.add_argument('--acc_type', type=int, default=1, help='accumulation type')

    parser.add_argument('--debug', type=bool, default=False, help='debug')
    args = parser.parse_args()

    lo_method = LocalOrientationEstimation.lo_methods(args.lo_method)
    params = dict(filename=args.filename, output_dir=args.output_dir, new_shape=args.new_shape,
                  fft_peak_th=args.fft_peak_th, block_width_size=args.block_width_size,
                  block_height_size=args.block_height_size, block_overlap=args.block_overlap, lo_method=lo_method,
                  lo_certainty_th=args.lo_certainty_th, peak_blur_sigma=args.peak_blur_sigma,
                  acc_type=args.acc_type,debug=args.debug)
    main(**params)

