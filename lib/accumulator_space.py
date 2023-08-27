"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
 version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
details.

You should have received a copy of the GNU Affero General Public License along with this program. If not,
see <http://www.gnu.org/licenses/>.

"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

from lib.geometry import Line
from lib.image import Color

class AccumulationSpace:

    def __init__(self, l_lo, output_dir, img, mask, type=0, debug=True):
        """
        Implementation of the accumulation space (algorithm 3 in paper). Two different implementations are defined depending of the value of
        type parameter. If type == 0, It is a matrix where each element is the number of lines that pass by
        that pixel. If type == 1, it is a matrix where each element is the sum of line intersection at that pixel.
        :param l_lo: list of lines
        :param output_dir: debugging output directory
        :param img: image matrix
        :param mask: background mask
        :param type: if 0, use estimation of intersection between lines. If 1, compute intersection between lines
        :param debug: debug flag
        """
        self.l_lo = l_lo
        self.output_dir = output_dir
        self.img = img
        self.mask = mask
        self.debug = debug
        self.type = type

    def compute_intersection(self, lo_i: Line, lo_j: Line, idx_i, idx_j):
        """
        Compute intersection between two lines. NOT USED
        :param lo_i: line i
        :param lo_j: line j
        :param idx_i: index of line i
        :param idx_j: index of line j
        :return:
        """
        # 1.0 Compute intersection between two lines
        x, y = lo_i.compute_intersection_with_line(lo_j)
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            return None

        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        if x < 0 or x >= self.img.shape[1] or y < 0 or y >= self.img.shape[0]:
            return None

        intersection = (x, y)

        if self.debug:
            # 2.0 Draw intersection point
            img_line = self.img.copy()
            img_line = lo_i.img_draw_line(img_line)
            img_line = lo_j.img_draw_line(img_line)
            cv2.circle(img_line, intersection, 5, Color.blue, -1)
            cv2.imwrite(str(Path(self.output_dir) / f'{idx_i}_{idx_j}.png'), img_line)


        return intersection[::-1]

    def debug_accumulation_space(self, accumulation_space):
        # Calculate the 2D histogram

        plt.figure()
        # Plot the histogram
        plt.imshow(accumulation_space, interpolation='nearest', cmap='hot')
        plt.xlabel('Pixel Intensity (Channel 1)')
        plt.ylabel('Pixel Intensity (Channel 2)')
        plt.title('2D Image Histogram')
        plt.colorbar()
        plt.savefig(str(Path(self.output_dir).parent / 'as.png'))
        plt.close()
        return

    def lines_pass_through_accumulation(self, l_lo):
        """
        Accumulate lines in the accumulation space. Every pixel that is in a line is incremented by 1
        :param l_lo: list of lines
        :return: accumulation space with the lines
        """
        accumulation_space = np.zeros(self.img.shape[:2], dtype=int)

        for i, lo_i in enumerate(l_lo):
            mask = np.zeros(accumulation_space.shape[:2], dtype=np.uint8)
            mask = lo_i.img_draw_line(mask, Color.gray_white, thickness=1)
            accumulation_space[mask == Color.gray_white] += 1

        return accumulation_space

    def lines_intersection_accumulation(self, l_lo):
        """
        Compute intersection between lines and accumulate them in the accumulation space.
        Acummulation space has same size (width and height) as the image
        :param l_lo: list of lines
        :return: accumulation space with the intersection of lines
        """
        accumulation_space = np.zeros(self.img.shape[:2], dtype=int)
        for i, lo_i in enumerate(l_lo):
            for j, lo_j in enumerate(l_lo):
                if i == j:
                    continue
                intersection = self.compute_intersection(lo_i, lo_j, i, j)
                if intersection is None:
                    continue
                accumulation_space[intersection] += 1

        return accumulation_space

    def run(self):
        """
        Compute accumulation space. Algorithm 3 in paper
        :return: accumulation space matrix
        """

        if self.type > 0:
            accumulation_space = self.lines_intersection_accumulation(self.l_lo)
        else:
            accumulation_space = self.lines_pass_through_accumulation(self.l_lo)

        ######################### Normalize accumulation space for visualization only
        normalized_accumulation_space = ((accumulation_space / accumulation_space.max())*255)
        normalized_accumulation_space = np.where(normalized_accumulation_space > 255, 255,
                                                 normalized_accumulation_space).astype(np.uint8)
        cv2.imwrite(f'{str(Path(self.output_dir).parent)}/accumulator.png', normalized_accumulation_space)

        if self.debug:
            self.debug_accumulation_space(accumulation_space)
        ############################################################################################################
        return accumulation_space
