"""
Implementation of the paper: "Pith Estimation on Rough Log End images using Local Fourier Spectrum Analysis" by Rudolf Schraml and Andreas Uhl.
http://dx.doi.org/10.2316/P.2013.797-012
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

from lib.image import resize_image_using_pil_lib, Color, Drawing, rgb2gray

class SplitImageInBlock:
    def __init__(self, block_width_size=15, block_height_size=15, block_overlap=0.0, mask=None, img=None,
                 output_dir=None, debug=False):
        """
        Split image in l_blocks
        :param block_width_size: block width size (px)
        :param block_height_size:  block width size (px)
        :param block_overlap: percentage of overlap between l_blocks. 0.0 means no overlap, 1.0 means 100% overlap
        :param mask: Disk mask to apply to image. If mask is not None, then only extract l_blocks that do not contain background.
        :param img: disk image
        :param output_dir: debugging output directory
        """
        #seed for reproducibility
        np.random.seed(0)
        self.overlap = block_overlap
        self.mask = mask
        self.img = img
        height, width, _ = self.img.shape
        #block width
        self.split_width = int(block_width_size) #int( width / width_partition)
        #block height
        self.split_height = int(block_height_size) #int (height / height_partition)
        self.output_dir = output_dir
        self.debug=debug


    @staticmethod
    def start_points(size, split_size, overlap=0):
        """
        Axis coordinates
        :param size: image axis size
        :param split_size: block size
        :param overlap: overlapping percentage
        :return: list of points axis coordinates
        """
        l_points = [0]
        stride = int(split_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                if split_size == size:
                    break
                #add point
                l_points.append(size - split_size)
                break
            else:
                #add point
                l_points.append(pt)
            counter += 1

        return l_points
    def get_blocks_top_left_coordinates(self):
        """
        Get top left coordinates of each block. Blocks coordinates are not aligned if overlap > 0
        :return: top left block coordinates list
        """
        img_h, img_w, _ = self.img.shape
        X_points = self.start_points(img_w, self.split_width, self.overlap)
        Y_points = self.start_points(img_h, self.split_height, self.overlap)

        if self.overlap > 0:
            # Noise is added to the index of each block to avoid block alignment
            l_coordinates = [(self.add_noise_to_coordinates(i, self.img.shape[0]),
                    self.add_noise_to_coordinates(j, self.img.shape[1])) for i in Y_points for j in X_points]
        else:
            l_coordinates = [(i,j) for i in Y_points for j in X_points]


        return l_coordinates

    def add_noise_to_coordinates(self, idx, higher, noise = 3):
        """
        Add noise to block coordinates
        :param idx: index
        :param higher: max index
        :param noise: normal distribution noise
        :return: idx with noise
        """
        idx_noise = np.round(np.random.normal(loc=idx, scale=noise)).astype(int)
        idx_noise = np.where(idx_noise < 0, 0, idx_noise)
        idx_noise = np.where(idx_noise > higher - 1, higher - 1, idx_noise)
        return int(idx_noise)
    def extract_path(self,i,j,img):
        """
        Extract block from image
        :param i: top left y coordinate
        :param j: top left x coordinate
        :param img: image
        :return: patch of size (split_height, split_width)
        """
        return img[i:i + self.split_height, j:j + self.split_width]

    def extract_blocks(self):
        """
        Extract l_blocks from image. If mask is not None, then only extract l_blocks that do not contain background.
        :return:
        """
        l_full_block_coordinates = self.get_blocks_top_left_coordinates()
        l_blocks = []
        l_filtered_blocks_coordinates = []

        for i_noise, j_noise in l_full_block_coordinates:
            split = self.extract_path(i_noise, j_noise, self.img)
            if self.mask is not None:
                split_mask = self.extract_path(i_noise, j_noise, self.mask)
                background_in_block = 0 in np.unique(split_mask).tolist()
                if background_in_block:
                    continue

            l_blocks.append(split)
            l_filtered_blocks_coordinates.append((i_noise, j_noise))

        return l_blocks, l_filtered_blocks_coordinates


    def save_img_blocks(self):
        cv2.imwrite(f"{self.output_dir}/img.png", self.img)
        cv2.imwrite(f"{self.output_dir}/mask.png", self.mask)
        if self.debug:
            blocks_dir = Path(self.output_dir) / "l_blocks"
            blocks_dir.mkdir(parents=True, exist_ok=True)
            [cv2.imwrite(f"{str(blocks_dir)}/block_nro_{idx}.png", block) for idx, block in enumerate(self.l_blocks)]


    def run(self):
        self.l_blocks, self.l_coordinates = self.extract_blocks()
        return



class LocalOrientationEstimation:
    peak = 0
    lsr = 1
    wlsr = 2
    pca = 3
    def __init__(self, img, mask, l_blocks,l_coordinates, type, certainty_threshold = 0.7, fft_peak_th=0.6, output_dir=None,
                 debug = True):
        """
        Local Orientation Estimation. Section 2 of the paper
        "Pith Estimation on Rough Log End images using Local Fourier Spectrum Analysis" by Rudolf Schraml and Andreas Uhl.
        :param img: RGB image
        :param mask: mask of the image
        :param l_blocks: list of image patches
        :param l_coordinates: list of top left coordinates of each block
        :param type: type of local orientation estimation
        :param certainty_threshold:
        :param output_dir: debug output directory
        :param debug: debug flag
        """
        self.img = img
        self.mask = mask
        self.output_dir = output_dir
        self.l_blocks = l_blocks
        self.l_coordinates = l_coordinates
        self.debug = debug
        self.type = type
        self.certainty_threshold = certainty_threshold
        self.fft_peak_th = fft_peak_th

    def compute_fourier_spectrum(self, l_blocks):
        """
        Compute Fourier Spectrum Magnitude of each block
        :param l_blocks: list of image l_blocks
        :return: list of Fourier Spectrum Magnitude of each block
        """
        l_fs_blocks = []
        for block in l_blocks:
            block = rgb2gray(block)
            f = np.fft.fft2(block)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            l_fs_blocks.append(magnitude_spectrum)

        if self.debug:
            #save to disk each block
            lof_dir = Path(self.output_dir) / "fourier_spectrum"
            lof_dir.mkdir(exist_ok=True, parents=True)
            [cv2.imwrite(f"{str(lof_dir)}/fs_block_nro_{idx}.png", block) for idx, block in enumerate(l_fs_blocks)]

        return l_fs_blocks

    def compute_the_two_highest_peaks(self, fs):
        """
        Compute the two highest peaks of the Fourier Spectrum Magnitude
        :param fs: Fourier Spectrum Magnitude
        :return:
        """
        fs = fs.copy()
        max_pos_rel, first_highest_value = self.compute_highest_peak_position(fs)
        fs[max_pos_rel[1], max_pos_rel[0]] = 0
        max_pos_rel_2, second_highest_value = self.compute_highest_peak_position(fs)
        return first_highest_value, second_highest_value

    def thresholding_frequencies_with_high_magnitude(self, fs, ratio_threshold = 0.6, lamb = 0.8):
        """
        Thresholding frequencies with high magnitude.
        :param fs: Fourier Spectrum Magnitude
        :param ratio_threshold: ratio threshold to consider the second-highest peak significant
        :param lamb: threshold parameter to filter out frequencies with low magnitude
        :return: filtered Fourier Spectrum Magnitude
        """
        fs = fs.copy()
        # Get the two highest peaks
        first_highest_value, second_highest_value = self.compute_the_two_highest_peaks(fs)
        ratio = second_highest_value / first_highest_value
        if ratio < ratio_threshold:
            # the second-highest peak is not significant, so we keep only the highest peak frequency
            fs = fs * (fs == first_highest_value)
        else:
            T = lamb * first_highest_value
            fs = fs * (fs > T)
        return fs

    def compute_band_filtering(self, fs):
        height, width = fs.shape
        # Band-pass filter frequencies. Allowed frequencies are between 1/64 and 1/3 of the image height
        band_pass_low_frequency = height / 64
        band_pass_high_frequency = height / 3
        center = (int(width / 2), int(height / 2))
        radius = int(band_pass_high_frequency)
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        radius = int(band_pass_low_frequency)
        cv2.circle(mask, center, radius, 0, -1)
        return cv2.bitwise_and(fs, fs, mask=mask)

    def preprocess_fourier_spectrum(self, l_fs_blocks):
        """
        Preprocessing of fourier spectrum. Section 2.3.1 of paper.
        :param l_fs_blocks: list of fourier spectrum magnitude of each block
        :return: list of preprocessed fourier spectrum magnitude of each block
        """
        # Band-pass filter. Paper do not mention the size of the band-pass filter
        if len(l_fs_blocks) == 0:
            return []
        height, width = l_fs_blocks[0].shape
        # Band-pass filter frequencies. Allowed frequencies are between 1/64 and 1/3 of the image height
        band_pass_low_frequency = height / 64
        band_pass_high_frequency = height / 3
        center = (int(width / 2), int(height / 2))
        radius = int(band_pass_high_frequency)
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        radius = int(band_pass_low_frequency)
        cv2.circle(mask, center, radius, 0, -1)

        # Apply band-pass filter
        l_pre_fs_blocks = [self.compute_band_filtering(blocks) for blocks in l_fs_blocks]
        # Thresholding frequencies with high magnitude.
        l_pre_fs_blocks = [self.thresholding_frequencies_with_high_magnitude(blocks, self.fft_peak_th) for blocks in l_pre_fs_blocks]

        return l_pre_fs_blocks

    @staticmethod
    def compute_highest_peak_position(block):
        """
        Compute the position of the highest peak of block
        :param block: image block
        :return: max position relative to the block and the value of the highest peak
        """
        max_pos_rel = np.unravel_index(np.argmax(block, axis=None), block.shape)[::-1]
        return max_pos_rel, block[max_pos_rel[1], max_pos_rel[0]]
    def lo_linear_symmetry_peak_analysis(self, l_pre_fs_blocks):
        """
        Section 2.3.2. For each fft block, split by half and select the position of the highest peak. Then, compute
        the line that passes through the center of the block and the peak position.
        :param l_pre_fs_blocks: list of preprocessed fourier spectrum magnitude of each block
        :return: list of lines of linear symmetry
        """
        l_lo = []
        for idx, block in enumerate(l_pre_fs_blocks):
            height, width = block.shape
            #split block by half
            left_block = block[:, :int(width/2)+1]
            # compute the position of the highest peak
            max_pos_rel,_ = self.compute_highest_peak_position(left_block)
            # compute the line that passes through the center of the block and the peak position
            center_rel = (int(width/2), int(height/2))
            i, j = self.l_coordinates[idx]
            center_abs = (j + int(width/2), i + int(height/2))
            max_pos_abs = (j + max_pos_rel[0], i + max_pos_rel[1])
            line = Line(center_rel, max_pos_rel, center_abs, max_pos_abs, 1)
            l_lo.append(line)

        return l_lo

    @staticmethod
    def compute_least_squares(A,y):
        # A^T A x = A^T y
        ATA = np.dot(A.T, A)
        invATA = np.linalg.inv(ATA)
        # x = (A^T A)^-1 A^T y
        x = np.dot(invATA, A.T).dot(y)
        return x

    def least_squares_not_independent_term(self, x, y, epsilon=1e-6):
        # compute the least squares solution of Ax = y. Using matrix operations
        if np.unique(x).size == 1 and np.unique(y).size > 1:
            # if all x are the same, then the slope is infinite
            return np.inf

        #Independent term is discarted because data is centered. Independent term is always 0 (or close to 0)
        # A = [x 1]
        A = np.vstack([x, np.ones(len(x))]).T
        # x = [a b]
        # y = [y]
        m, n = self.compute_least_squares(A, y)
        # m, n = np.linalg.lstsq(A, y, rcond=None)[0]
        # assert np.abs(m_m - m)< epsilon and np.abs(n_m - n) < epsilon
        assert n < epsilon
        return m

    def compute_line_given_coefficient(self, a, width, height, idx, certainty):
        """
        Compute the line given the coefficient a relative to the block center
        :param a: line slope
        :param width: block width
        :param height: block height
        :param idx: index of the block
        :param certainty: certainty of the estimation
        :return: line
        """
        p1_rel = (width / 2, height / 2)
        X = 1
        y = a * X
        if np.abs(a) == np.inf:
            p2_rel = (width / 2, 0)

        elif y > (height)/2:
            p2_rel = (X+width / 2, height-1) #Vertical line passing by the block center

        elif y < -1*(height/2):
            p2_rel = (X+width / 2, 0)

        elif a == 0:
            p2_rel = (0, height / 2)

        else:
            p2_rel = np.array([X, y]) + np.array([width / 2, height / 2]) #Block origin is the corner (0,0)

        i, j = self.l_coordinates[idx]
        change_base_vector = np.array([j, i])
        p2 = p2_rel + change_base_vector
        p1 = p1_rel + change_base_vector

        line = Line(p1_rel, p2_rel, p1, p2, certainty)
        return line

    @staticmethod
    def certainty_lsq(X, Y):
        """
        Compute the coefficient of determination (R^2) of the least squares solution. Defined in section 2.3.3
        :param X: Independent variable
        :param Y: Dependent variable
        :return: R^2
        """
        corr_matrix = np.corrcoef(X, Y)
        corr = corr_matrix[0, 1]
        R_sq = corr ** 2
        if np.isnan(R_sq):
            R_sq = 0

        return R_sq


    def debug_lsqr(self, X,Y, magnitude, a, line, idx):
        # Plot scatter dots with color based on magnitude vector
        plt.figure()
        plt.scatter(X, Y, c=magnitude)
        plt.colorbar()
        X.sort()
        if 0 < np.abs(a) < np.inf:
            line_values = a * X
            x = X
        elif a == np.inf:
            # vertical line
            x = [0, 0]
            line_values = [np.min(Y), np.max(Y)]
        else:
            # a == 0
            line_values = np.zeros_like(X)
            x = X
        plt.plot(x, line_values, 'r')
        # invert y-axis
        plt.gca().invert_yaxis()
        plt.title(f"LSR certanty:{line.certainty:.2f}. Coeficient: {a:.2f}")
        plt.savefig(f"{self.lof_dir}/block_nro_{idx}_lsr.png")
        plt.close()
        return

    @staticmethod
    def get_data_points_from_fft(fs_block):
        """
        Get the data points from the fft block to fit the line by Least Squares/PCA. Defined in section 2.3.3
        of Reference Paper
        :param fs_block: fft magnitude block
        :return: data points, magnitude, height and width of the block
        """
        Y, X = np.nonzero(fs_block)
        magnitude = fs_block[Y, X]
        height, width = fs_block.shape
        Y -= int(height/2)
        X -= int(width/2)
        return X, Y, magnitude, height, width

    def least_squares_weighted(self, X, Y, magnitude):
        """
        Weighted least squares
        :param X: Independent variable
        :param Y: Dependent variable
        :param magnitude: Magnitude of the fft data |fft(X,Y)|^2
        :return:
        """
        # compute the least squares solution of Ax = y
        if np.unique(X).size == 1 and np.unique(Y).size > 1:
            # if all x are the same, then the slope is infinite
            return np.inf

        # A = [x 1]
        A = np.vstack([X, np.ones(len(X))]).T
        # x = [a b]
        # y = [y]
        w = np.sqrt(magnitude)
        #m, n = np.linalg.lstsq(A * w[:, np.newaxis], Y * w, rcond=None)[0]
        m, n = self.compute_least_squares(A * w[:, np.newaxis] , Y * w)
        assert n < 10**-5
        return m
    def lo_linear_symmetry_lsr(self, pre_fs_blocks, weighted=False):
        """
        Compute the lines of symmetry using Least Squares Regression. Defined in section 2.3.3 and 2.3.4
        of Reference Paper
        :param pre_fs_blocks: fourier magnitude blocks
        :param weighted: flag to use weighted least squares
        :return: list of lines of symmetry
        """
        l_lo = []
        for idx, fs_block in enumerate(pre_fs_blocks):
            # Get coordinates of non-zero pixels
            X, Y, magnitude, height, width = self.get_data_points_from_fft(fs_block)
            # Fit a line, Y = aX, through some noisy data-points
            a = self.least_squares_not_independent_term(X, Y) if not weighted else self.least_squares_weighted(X, Y, magnitude)
            certainty = self.certainty_lsq(X, Y)
            line = self.compute_line_given_coefficient(a, width, height, idx, certainty)
            l_lo.append(line)

            if self.debug:
                self.debug_lsqr(X, Y, magnitude, a, line, idx)

        return l_lo

    def return_type_str(self):
        if self.type == self.peak:
            return "peak"
        elif self.type == self.lsr:
            return "lsr"
        elif self.type == self.wlsr:
            return "wlsr"
        elif self.type == self.pca:
            return "pca"
        else:
            raise Exception("Invalid type of symmetry")

    def generating_debug_images(self, l_lo, name='all_lines', debug=True):
        img_blocks = self.img.copy()
        img_all_lines = self.img.copy()
        for idx, line in enumerate(l_lo):
            height, width, _ = self.l_blocks[idx].shape
            top_left = line.p1 - np.array([int(width / 2), int(height / 2)])
            bottom_right = line.p1 + np.array([int(width / 2), int(height / 2)])
            img_blocks = Drawing.rectangle(img_blocks, top_left.astype(int), bottom_right.astype(int), Color.black, 2)
            img_all_lines = line.img_draw_line(img_all_lines, thickness=1)
            radius = 3
            color = Color.blue
            thickeness = -1
            img_all_lines = Drawing.circle(img_all_lines, line.p1.astype(int), thickeness, color, radius)


        cv2.imwrite(f"{str(self.lof_dir.parent.parent)}/img_{name}.png", img_all_lines)
        cv2.imwrite(f"{str(self.lof_dir.parent.parent)}/img_blocks_{name}.png", img_blocks)


        if debug:
            #save to disk each block
            for idx, line in enumerate(l_lo):
                fs_block = line.block_draw_line(self.pre_fs_blocks[idx], extended=False)
                fs_block_line = line.block_draw_line(self.pre_fs_blocks[idx])
                radius = 2
                color = Color.blue
                thickeness = -1
                fs_block_line = Drawing.circle(fs_block_line, line.p1_rel.astype(int), thickeness, color, radius)
                fs_block_line = Drawing.circle(fs_block_line, line.p2_rel.astype(int), thickeness, color, radius)

                cv2.imwrite(f"{str(self.lof_dir)}/block_nro_{idx}_all.png",
                            np.hstack(
                                (
                                    fs_block,
                                    np.hstack(
                                        (
                                            fs_block_line , line.block_draw_line(self.l_blocks[idx])
                                        )
                                    )
                                )

                            )
                            )
                img_line = line.img_draw_line(self.img)
                img_all_lines = line.img_draw_line(img_all_lines)
                radius = 3
                color = Color.blue
                thickeness = -1
                img_line = Drawing.circle(img_line, line.p1.astype(int), thickeness, color, radius)
                #draw rectangle using opencv
                height, width, _ = self.l_blocks[idx].shape
                top_left = line.p1 - np.array([ int(width/2), int(height/2)])
                bottom_right = line.p1 + np.array([ int(width/2), int(height/2)])
                thickeness = 2
                img_line = Drawing.rectangle(img_line, top_left.astype(int), bottom_right.astype(int), Color.black, thickeness)
                #img_all_lines = Drawing.rectangle(img_all_lines, top_left, bottom_right, Color.black, thickeness)
                cv2.imwrite(f"{str(self.lof_dir)}/block_nro_{idx}_img.png", img_line)
                x,y = line.p2_rel.astype(int)
                fs_block[y,x,:] = Color.blue
                cv2.imwrite(f"{str(self.lof_dir)}/block_nro_{idx}_peak.png", fs_block)
        return

    @staticmethod
    def compute_pca_analysis(X, Y):
        """
        Compute PCA analysis using numpy
        :param X: Independent variable
        :param Y: Dependent variable
        :return: Eigenvectors and eigenvalues
        """
        data = np.vstack([X, Y]).T
        pca = PCA(n_components=2)
        pca.fit(data)
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_
        return eigenvectors, eigenvalues

    def debug_pca(self,X, Y, a, eigenvalues, eigenvectors, certainty, idx):
        plt.figure()
        plt.scatter(X, Y)
        Xextended = np.array(X.tolist())
        Xextended.sort()

        origin = np.array([0, 0])
        plt.plot(Xextended, a * Xextended, 'r')
        for eigenvalue, component in zip(eigenvalues, eigenvectors):
            # add arrow lenght
            plt.arrow(
                origin[0],
                origin[1],
                component[0],
                component[1],
                head_width=0.2,
                head_length=0.2,
                fc='k',
                ec='k'
            )

        plt.gca().invert_yaxis()
        plt.title(
            f"PCA. certainty: {certainty:.2f}. coeff: {a:.2f}\n lamb1={eigenvalues[0]:.2f}, lamb2={eigenvalues[1]:.2f}")
        plt.savefig(f"{self.lof_dir}/block_nro_{idx}_pca.png")
        plt.close()
        return
    def lo_linear_symmetry_pca(self, pre_fs_blocks):
        """
        Compute the lines of symmetry using PCA. Section 3.2.5 of Reference Paper
        :param pre_fs_blocks: list of pre_fs_blocks
        :return: list of lines of symmetry
        """
        l_lo = []
        for idx, fs_block in enumerate(pre_fs_blocks):
            X, Y, _, height, width = self.get_data_points_from_fft(fs_block)
            # Compute PCA analysis using numpy
            if X.shape[0]>0:
                eigenvectors, eigenvalues = self.compute_pca_analysis(X, Y)
                # extract line direction from principal components
                u_p = eigenvectors[0] #first principal component
                a = u_p[1]/u_p[0] if u_p[0] != 0 else np.inf
                certainty = (eigenvalues[0] - eigenvalues[1] ) / eigenvalues[0] if eigenvalues[0] > 0 else 0
            else:
                a = 0
                certainty = 0
            line = self.compute_line_given_coefficient(a, width, height, idx, certainty)
            l_lo.append(line)

            if self.debug:
                self.debug_pca(X, Y, a, eigenvalues, eigenvectors, certainty, idx)


        return l_lo

    @staticmethod
    def lo_methods(method):
        if method == 'pca':
            return LocalOrientationEstimation.pca
        elif method == 'lsr':
            return LocalOrientationEstimation.lsr

        elif method == 'wlsr':
            return LocalOrientationEstimation.wlsr

        elif method == 'peak':
            return LocalOrientationEstimation.peak
        else:
            raise Exception(f"Invalid method {method}")

    def lo_linear_symmetry(self, pre_fs_blocks):
        self.lof_dir = Path(self.output_dir) / f"lo_linear_symmetry_{self.return_type_str()}_analysis"
        if self.debug:
            self.lof_dir.mkdir(exist_ok=True, parents=True)

        if self.type == self.peak:
            l_lo = self.lo_linear_symmetry_peak_analysis(pre_fs_blocks)

        elif self.type == self.lsr:
            l_lo = self.lo_linear_symmetry_lsr(pre_fs_blocks)

        elif self.type == self.wlsr:
            l_lo = self.lo_linear_symmetry_lsr(pre_fs_blocks, weighted=True)

        elif self.type == self.pca:
            l_lo = self.lo_linear_symmetry_pca(pre_fs_blocks)

        else:
            raise Exception("Invalid type of symmetry")

        return l_lo

    def filter_lo_by_certainty(self, l_lo):
        return [line for line in l_lo if line.certainty >= self.certainty_threshold]
    def run(self):
        """
        Section 2.3 of Reference Paper
        :return: list of lines of symmetry
        """
        fs_blocks = self.compute_fourier_spectrum(self.l_blocks)
        self.pre_fs_blocks = self.preprocess_fourier_spectrum(fs_blocks)
        l_lo = self.lo_linear_symmetry(self.pre_fs_blocks)
        self.generating_debug_images(l_lo, debug=self.debug)
        l_lo = self.filter_lo_by_certainty(l_lo)
        self.generating_debug_images(l_lo, name="filtered_lines", debug=False)
        return l_lo

class Line:
    def __init__(self, p1_rel, p2_rel, p1, p2, certainty):
        """
        Line of symmetry. Line equation: a*x+b*y+c=0
        :param p1_rel: point 1 relative to the block. Pixel coordinates
        :param p2_rel: point 2 relative to the block. Pixel coordinates
        :param p1: point 1 absolute coordinates. Pixel
        :param p2: point 2 absolute coordinates. Pixel.
        :param certainty: certainty of the line of symmetry estimation
        """
        self.p1_rel = np.array(p1_rel)
        self.p2_rel = np.array(p2_rel)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.certainty = certainty


    def compute_line_coefficients(self, p1, p2):
        """
        a*x+b*y+c=0
        """
        x1, y1 = p1.ravel()
        x2, y2 = p2.ravel()

        if x2 == x1:
            # Vertical line: x = c
            a = 1
            b = 0
            c = -x1
        elif y1 == y2:
            # Horizontal line: y = c
            a = 0
            b = 1
            c = -y1
        else:
            a = y1-y2
            b = x2-x1
            c = x2*y1-x1*y2

        return a, b, c


    def compute_intersection_with_line(self, line):
        """
        Compute the intersection between two lines
        :param line:
        :return:
        """
        a1, b1, c1 = self.compute_line_coefficients(self.p1, self.p2)
        a2, b2, c2 = line.compute_line_coefficients(line.p1, line.p2)
        # Compute the intersection between two lines
        determinant = a1 * b2 - a2 * b1
        if determinant == 0:
            # No intersection
            return None, None
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return x, y

    def get_line_coordinates(self,a,b,c,x=None,y=None):
        if x is None and y is None:
            return None, None

        if a == 0 and b==0 :
            return None, None

        if a == 0 and b != 0:
            y = c/b
            return 0,y

        if b==0 and a != 0:
            x = c/a
            return x,0

        if x is not None:
            y = (c-a*x)/b
            return x,y

        if y is not None:
            x = (c-b*y)/a
            return x,y

        return None, None
    def compute_intersection_with_block_boundaries(self, p1, p2, img):
        # Get the image dimensions
        if img.ndim == 3:
            height, width, _ = img.shape
        else:
            height, width = img.shape
        a, b, c = self.compute_line_coefficients( p1, p2)
        if b==0 and np.abs(a)>0:
            # Vertical line
            x = int(p2[0])
            x1, y1 = x, 0
            x2, y2 = x, height - 1
            return x1, y1, x2, y2, None, None, None, None
        if a==0 and np.abs(b)>0:
            # Horizontal line
            y = int(p2[1])
            x1, y1 = 0, y
            x2, y2 = width - 1, y
            return x1, y1, x2, y2, None, None, None, None
        x1, y1 = 0, None
        x2, y2 = width - 1, None
        x3, y3 = None, 0
        x4, y4 = None, height - 1
        x1,y1 = self.get_line_coordinates(a,b,c,x1,y1)
        x2,y2 = self.get_line_coordinates(a,b,c,x2,y2)
        x3,y3 = self.get_line_coordinates(a,b,c,x3,y3)
        x4,y4 = self.get_line_coordinates(a,b,c,x4,y4)

        return x1, y1, x2, y2, x3, y3, x4, y4

    def draw_line(self, x1, y1, x2, y2, block, thickness, color):
        # Draw the line
        integer = lambda x: int(np.round(x))
        return cv2.line(block.copy(), (integer(x1), integer(y1)), (integer(x2), integer(y2)), color, thickness)

    def block_draw_line(self, block, color = Color.red, thickness = 2, extended = True):

        # convert binary block array to rgb
        if len(block.shape) == 2:
            block = cv2.cvtColor(block.astype(np.uint8), cv2.COLOR_GRAY2RGB).astype(np.uint8)

        if not extended:
            return block.copy()

        # Calculate the intersections with the image boundaries
        x1, y1, x2, y2, x3, y3, x4, y4 = self.compute_intersection_with_block_boundaries(self.p1_rel, self.p2_rel,
                                                                                                        block)
        # Draw the line on the image
        extended_line = self.draw_line(x1, y1, x2, y2, block, thickness, color)

        return extended_line


    def img_draw_line(self, img, color = Color.red, thickness = 2):
        # Calculate the intersections with the image boundaries
        x1, y1, x2, y2, x3, y3, x4, y4 = self.compute_intersection_with_block_boundaries(
            self.p1, self.p2, img)
        # Draw the line on the image
        extended_line = self.draw_line(x1, y1, x2, y2, img, thickness, color)

        return extended_line

class PithDetector:
    def __init__(self, img_in=None, mask=None, block_overlap=0.35, block_width_size=15, block_height_size=15,
                 fft_peak_th=0.6, lo_method=LocalOrientationEstimation.pca, lo_certainty_th=0.5, acc_type=0,
                 peak_blur_sigma=5, debug=False, output_dir=None):
        """
        Implementation of the pith detection algorithm described in Reference Paper
        :param img_in: input image
        :param mask: input background mask
        :param block_width_size: Pixel partition size in the width direction in order to get patches
        :param block_height_size: Pixel partition size in the height direction in order to get patches
        :param block_overlap: overlap between patches
        :param output_dir: output directory
        :param debug: debug flag
        :param lo_method: method to use for local orientation estimation
        :param lo_certainty_th: threshold for local orientation estimation
        :param acc_type: type of accumulator to use
        :param peak_blur_sigma: sigma for peak blurring
        """
        self.img = img_in
        self.mask = mask
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        self.lo_fft_peak_th = fft_peak_th
        self.lo_width_partition = block_width_size
        self.lo_height_partition = block_height_size
        self.lo_overlap = block_overlap
        self.debug = debug
        self.lo_method = lo_method
        self.lo_certainty_threshold = lo_certainty_th
        self.acc_type = acc_type
        self.peak_blur_sigma = peak_blur_sigma
    def local_orientation_estimation(self, debug=True):
        """
        Compute local orientation of the image.
        :param debug:
        :return:
        """
        # 1.0 compute image regions
        block_splitter = SplitImageInBlock(block_width_size=self.lo_width_partition,
                                           block_height_size=self.lo_height_partition, block_overlap=self.lo_overlap,
                                           mask=self.mask, img=self.img, output_dir=self.output_dir, debug=debug)
        block_splitter.run()
        block_splitter.save_img_blocks()
        l_blocks = block_splitter.l_blocks
        l_coordinates = block_splitter.l_coordinates

        # 2.0 compute local orientation of each region
        lo_dir = Path(self.output_dir) / 'local_orientation'
        if debug:
            lo_dir.mkdir(exist_ok=True, parents=True)
        lo = LocalOrientationEstimation(img=self.img, mask=self.mask, l_blocks=l_blocks, l_coordinates=l_coordinates,
                                        output_dir=str(lo_dir), debug=debug, type = self.lo_method,
                                        certainty_threshold=self.lo_certainty_threshold, fft_peak_th = self.lo_fft_peak_th )
        l_lo = lo.run()

        return l_lo # list of local orientation objects
    def accumulation_space(self, l_lo, acc_type, debug=True):
        # 3.0 compute accumulation space
        as_dir = Path(self.output_dir) / 'accumulation_space'
        if debug:
            as_dir.mkdir(exist_ok=True, parents=True)
        as_obj = AccumulationSpace(l_lo=l_lo, output_dir=str(as_dir), img = self.img, mask=self.mask, type=acc_type, debug=debug)

        return as_obj.run()

    def find_peak(self, m_accumulation_space, sigma = 3):

        ################################################################################################################
        ac_blur =  cv2.GaussianBlur(m_accumulation_space.astype(np.uint8), (sigma, sigma), 0) if sigma > 0 else m_accumulation_space
        # find max location
        max_val = ac_blur.max()
        yy, xx = np.where(ac_blur >= max_val)
        max_loc =(xx.mean().astype(int), yy.mean().astype(int))
        # draw circle
        img = self.img.copy()
        for y,x in zip(yy,xx):
            img = cv2.circle(img, (np.round(x).astype(int),np.round(y).astype(int)), 2, Color.red, -1)
        img = cv2.circle(img, max_loc, 2, Color.blue, -1)
        # save img
        cv2.imwrite(str(Path(self.output_dir) / 'peak.png'), img)

        return max_loc
    def run(self):
        """Implementations of Block Area Selection - BAS algorithm described in Reference Paper"""
        l_lo = self.local_orientation_estimation(debug=self.debug)
        m_accumulation_space = self.accumulation_space(l_lo, self.acc_type, debug=self.debug)
        peak = self.find_peak(m_accumulation_space, self.peak_blur_sigma)

        return peak
class AccumulationSpace:

    def __init__(self, l_lo, output_dir, img, mask, type=0, debug=True):
        """
        Implementation of the accumulation space. Two different implementations are defined dependind of the value of
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
    def run(self):
        """
        Compute accumulation space
        :return: accumulation space
        """
        accumulation_space = np.zeros(self.img.shape[:2], dtype=int)
        if self.type == 0:
            for i,lo_i in enumerate(self.l_lo):
                mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                mask = lo_i.img_draw_line(mask, Color.gray_white,thickness=1)
                accumulation_space[mask == Color.gray_white] += 1

        else:
            for i, lo_i in enumerate(self.l_lo):
                for j, lo_j in enumerate(self.l_lo):
                    if i == j:
                        continue
                    intersection = self.compute_intersection(lo_i, lo_j, i, j)
                    if intersection is None:
                        continue
                    accumulation_space[intersection] += 1

        normalized_accumulation_space = ((accumulation_space / accumulation_space.max())*255)
        normalized_accumulation_space = np.where(normalized_accumulation_space > 255, 255,
                                                 normalized_accumulation_space).astype(np.uint8)
        cv2.imwrite(f'{str(Path(self.output_dir).parent)}/accumulator.png', normalized_accumulation_space)

        if self.debug:
            self.debug_accumulation_space(accumulation_space)

        return accumulation_space
def shraml_uhl_peak_detector(filename, output_dir, new_shape=640, fft_peak_th=0.8, width_partition=10,
                             height_partition=10, block_overlap=0.25, lo_method=LocalOrientationEstimation.pca,
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
    # 2.0 segment image
    mask = np.where( img_in== 255, 0, 255).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.uint8)

    # 3.0 pith detector
    # 3.1 coarse pith detector
    pith_detector = PithDetector(img_in=img_in, mask=mask, block_overlap=block_overlap,
                                 block_width_size=width_partition, block_height_size=height_partition,
                                 fft_peak_th=fft_peak_th, lo_method=lo_method, lo_certainty_th=lo_certainty_th,
                                 acc_type=acc_type, peak_blur_sigma=peak_blur_sigma, debug=debug,
                                 output_dir=str(output_dir))
    peak_coarse = pith_detector.run()

    tf = time.time()
    # 4.0 save results
    convert_original_scale = lambda peak: (np.array(peak) * np.array([o_width/new_shape,o_height/new_shape])).tolist()
    data = {'coarse': convert_original_scale(peak_coarse), 'exec_time(s)':tf-to}
    #print(f"{output_dir.name}: {data}")
    df = pd.DataFrame(data)
    df.to_csv(str(output_dir / 'pith.csv'), index=False)

    return peak_coarse


import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pith detector')
    parser.add_argument('--filename', type=str, required=True, help='image filename')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')

    #method parameters
    parser.add_argument('--new_shape', type=int, default=1000, help='new shape')
    parser.add_argument('--fft_peak_th', type=float, default=0.8, help='fourier transform peak threshold')
    parser.add_argument('--width_partition', type=int, default=100, help='width partition size (px)')
    parser.add_argument('--height_partition', type=int, default=100, help='height partition size (px)')
    parser.add_argument('--block_overlap', type=float, default=0.2, help='block overlapping')
    parser.add_argument('--lo_method', type=str, default='pca', help='lo method')
    parser.add_argument('--lo_certainty_th', type=float, default=0.9, help='lo certainty threshold')
    parser.add_argument('--peak_blur_sigma', type=int, default=3, help='peak blur sigma')
    parser.add_argument('--acc_type', type=int, default=1, help='accumulation type')

    parser.add_argument('--debug', type=bool, default=False, help='debug')
    args = parser.parse_args()

    lo_method = LocalOrientationEstimation.lo_methods(args.lo_method)
    params = dict(filename=args.filename, output_dir=args.output_dir, new_shape=args.new_shape,
                  fft_peak_th=args.fft_peak_th, width_partition=args.width_partition,
                  height_partition=args.height_partition, block_overlap=args.block_overlap, lo_method=lo_method,
                  lo_certainty_th=args.lo_certainty_th, peak_blur_sigma=args.peak_blur_sigma,
                  acc_type=args.acc_type,debug=args.debug)
    shraml_uhl_peak_detector(**params)

