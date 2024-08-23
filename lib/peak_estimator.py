"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
import cv2
import numpy as np
from pathlib import Path

from lib.image import Color

def find_peak( m_accumulation_space: np.array, img_in: np.array, output_dir: str, sigma = 3):
    """
    Find the peak of the accumulation space
    :param m_accumulation_space: matrix of the accumulation space
    :param img_in: raw RGB input image
    :param output_dir: directory for debugging
    :param sigma: Gaussian kernel size
    :return: peak pixel location
    """
    # Section 2.4
    ac_blur =  cv2.GaussianBlur(m_accumulation_space.astype(np.uint8), (sigma, sigma), 0) if sigma > 0 else m_accumulation_space
    # find max location
    max_val = ac_blur.max()
    yy, xx = np.where(ac_blur >= max_val)
    max_loc =(xx.mean().astype(int), yy.mean().astype(int))

    # draw circle
    img = img_in.copy()
    for y, x in zip(yy, xx):
        img = cv2.circle(img, (np.round(x).astype(int), np.round(y).astype(int)), 2, Color.red, -1)
    img = cv2.circle(img, max_loc, 2, Color.blue, -1)

    # save img_in
    cv2.imwrite(str(Path(output_dir) / 'peak.png'), img)

    return max_loc