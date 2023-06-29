#!/bin/bash
set -e

# Read input parameters
input=$1
new_shape=$2
th_low=$3
th_high=$4
hsize=$5
wsize=$6
BIN=$7
HOME=$8



# Execute algorithm
python $BIN/main.py --input $input --cx $Cx --cy $Cy --root $BIN --output_dir ./  --th_high $th_high --th_low $th_low \
  --hsize $hsize --wsize $wsize --sigma $sigma --save_imgs 1

