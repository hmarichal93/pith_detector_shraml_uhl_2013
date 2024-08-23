#!/bin/bash
set -e

# Read input parameters
input=$1
output_dir=$2
new_shape=$3
width_partition=$4
height_partition=$5
block_overlap=$6
fft_peak_th=$7
lo_method=$8
certainty_th=$9
acc_type=${10}
peak_blur_sigma=${11}
bin=${12}


# Execute algorithm
python $bin/main.py --filename $input --output_dir ./ --new_shape $new_shape --block_width_size $width_partition --block_height_size $height_partition --block_overlap $block_overlap --fft_peak_th $fft_peak_th --lo_method $lo_method --lo_certainty_th $certainty_th --acc_type $acc_type --peak_blur_sigma $peak_blur_sigma

