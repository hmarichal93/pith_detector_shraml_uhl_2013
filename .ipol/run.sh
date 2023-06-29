#!/bin/bash
set -e

# Read input parameters
input=$1
output_dir=$2
new_shape=$3
width_partition=$3
height_partition=$4
block_overlap=$5
fft_peak_th=$6
lo_method=$7
certainty_th=$8
acc_type=$9
peak_blur_sigma=${10}
bin=${11}

echo "ACA estoy"

# Execute algorithm
python $bin/main.py --input $input --output_dir $output_dir --new_shape $new_shape --width_partition $width_partition --height_partition $height_partition --block_overlap $block_overlap --fft_peak_th $fft_peak_th --lo_method $lo_method --certainty_th $certainty_th --acc_type $acc_type --peak_blur_sigma $peak_blur_sigma

