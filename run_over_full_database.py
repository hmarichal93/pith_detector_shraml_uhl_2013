import numpy as np
from pathlib import Path
import pandas as pd
import json

from main import shraml_uhl_peak_detector, LocalOrientationEstimation


def load_json(filepath: str) -> dict:
    """
    Load json utility.
    :param filepath: file to json file
    :return: the loaded json as a dictionary
    """
    with open(str(filepath), 'r') as f:
        data = json.load(f)
    return data


def write_json(dict_to_save: dict, filepath: str) -> None:
    """
    Write dictionary to disk
    :param dict_to_save: serializable dictionary to save
    :param filepath: path where to save
    :return: void
    """
    with open(str(filepath), 'w') as f:
        json.dump(dict_to_save, f)

def run_method_over_dataset(output_dir=None, params=None):
    root_imageset = '/data/maestria/datasets/cross-section/UruDendro_2019'
    metadata_filename = f'{root_imageset}/dataset_ipol.csv'
    root_images_dir = f'{root_imageset}/images/segmented'
    metadata = pd.read_csv(metadata_filename)
    data = pd.DataFrame(columns=['block','c_y','c_x', 'f_y', 'f_x', 'gt_y', 'gt_x', 'diff_f', 'diff_c', 'exec_time(s)'])
    distance = lambda x,y: np.linalg.norm(x - y)
    for idx, row in metadata.iterrows():

        output_img_dir = Path(output_dir) / row.Imagen
        output_img_dir.mkdir(parents=True, exist_ok=True)
        img_filename = f"{root_images_dir}/{row.Imagen}.png"
        shraml_uhl_peak_detector(img_filename, output_img_dir, **params)
        df = pd.read_csv(f"{output_img_dir}/pith.csv")

        centro_gt = np.array([row['cy'], row['cx']])
        centro_dt_c = df.coarse.values
        diff_c = distance(centro_gt, centro_dt_c)
        centro_dt_f = df.fine.values
        diff_f = distance(centro_gt, centro_dt_f)
        row = {'block': row.Imagen,'c_y': centro_dt_c[1], 'c_x': centro_dt_c[0], 'f_y': centro_dt_f[1], 'f_x': centro_dt_f[0], 'gt_y': centro_gt[1],
               'gt_x': centro_gt[0], 'diff_f': diff_f , 'diff_c': diff_c, 'exec_time(s)': df['exec_time(s)'].values[0]}
        data = pd.concat([data, pd.DataFrame([row])], ignore_index=True) \
            if data.shape[0] > 0 else pd.DataFrame(row, index=[0])

    output_csv =  f"{output_dir}/results.csv"
    data.to_csv(output_csv)
    print(f"Config: {output_csv.split('/')[-2]} - ExecTime(average): {data['exec_time(s)'].mean():.02f} - "
          f"Diff_c(average): {data['diff_c'].mean():.02f} - Diff_f(average): {data['diff_f'].mean():.02f}.")
import argparse


def optimization():
     l_fine_windows_size = [4,6,8]
     l_coarse_width_partition = [10,20]
     l_overlap = [0, 0.25, 0.5]
     l_coarse_lo_certainty = [ 0.6, 0.75, 0.9]
     accumulator_type = [0, 1]
     sigma_peak_blur = [3]
     counter = 0
     min_score = np.inf
     for fine_windows_size in l_fine_windows_size:#3
         for coarse_width_partition in l_coarse_width_partition:#2
             coarse_height_partition = coarse_width_partition
             for coarse_lo_certainty_threshold in l_coarse_lo_certainty:#3
                 for fine_lo_certainty_threshold in l_coarse_lo_certainty:#3
                     for fine_width_partition in l_coarse_width_partition:#2
                         fine_height_partition = fine_width_partition
                         for overlap in l_overlap:#3
                             for acc_type in accumulator_type:#2
                                 for sigma in sigma_peak_blur:#1
                                     params = dict(fine_windows_size = fine_windows_size,coarse_width_partition = coarse_width_partition,
                                        coarse_height_partition = coarse_height_partition, coarse_overlap = overlap, coarse_acc_type=acc_type,
                                        fine_overlap = overlap, coarse_lo_certainty_threshold=coarse_lo_certainty_threshold,
                                        coarse_peak_blur_sigma=sigma, fine_lo_certainty_threshold=fine_lo_certainty_threshold,
                                        fine_width_partition=fine_width_partition, fine_height_partition=fine_height_partition,
                                        fine_acc_type=acc_type, fine_peak_blur_sigma=sigma, debug=False)
                                     output_dir = f'/data/maestria/resultados/centro/shraml_uhl_2013_optimization/{counter}'
                                     Path(output_dir).mkdir(parents=True, exist_ok=True)
                                     results_file= f'{output_dir}/results.csv'
                                     #if Path(results_file).exists():
                                     #    counter += 1
                                     #    continue

                                     #write_json(params, f'{output_dir}/params.json')
                                     #run_method_over_dataset(output_dir, params)
                                     df = pd.read_csv(results_file)
                                     score = df['diff_c'].mean()
                                     if score < min_score:
                                         min_score = score
                                         print(f"New best score: Coarse Config {counter} {min_score} ")

                                     score = df['diff_f'].mean()
                                     if score < min_score:
                                         min_score = score
                                         print(f"New best score: Fine Config {counter} {min_score} ")

                                     counter += 1

     print(f"Best score: {min_score} ")
     return

if __name__=="__main__":
    # parser = argparse.ArgumentParser(description='Pith detector')
    # #parser.add_argument('--filename', type=str, required=True, help='image filename')
    # parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    #
    # #method parameters
    # parser.add_argument('--new_shape', type=int, default=640, help='new shape')
    # parser.add_argument('--fine_windows_size', type=int, default=4, help='fine windows size')
    # parser.add_argument('--coarse_width_partition', type=int, default=15, help='coarse width partition')
    # parser.add_argument('--coarse_height_partition', type=int, default=15, help='coarse height partition')
    # parser.add_argument('--coarse_overlap', type=float, default=0.25, help='coarse overlap')
    # parser.add_argument('--coarse_lo_method', type=str, default='pca', help='coarse lo method')
    # parser.add_argument('--coarse_lo_certainty_threshold', type=float, default=0.7, help='coarse lo certainty threshold')
    # parser.add_argument('--coarse_peak_blur_sigma', type=int, default=3, help='coarse peak blur sigma')
    # parser.add_argument('--fine_width_partition', type=int, default=20, help='fine width partition')
    # parser.add_argument('--fine_height_partition', type=int, default=20, help='fine height partition')
    # parser.add_argument('--fine_overlap', type=float, default=0.2, help='fine overlap')
    # parser.add_argument('--fine_lo_method', type=str, default='pca', help='fine lo method')
    # parser.add_argument('--fine_lo_certainty_threshold', type=float, default=0.5, help='fine lo certainty threshold')
    # parser.add_argument('--fine_peak_blur_sigma', type=int, default=3, help='fine peak blur sigma')
    # parser.add_argument('--debug', type=bool, default=False, help='debug')
    # args = parser.parse_args()
    # coarse_lo_method = LocalOrientationEstimation.lo_methods(args.coarse_lo_method)
    # fine_lo_method = LocalOrientationEstimation.lo_methods(args.fine_lo_method)
    #
    # params = dict(new_shape=args.new_shape, fine_windows_size=args.fine_windows_size,
    #               coarse_width_partition=args.coarse_width_partition, coarse_height_partition=args.coarse_height_partition,
    #               coarse_overlap=args.coarse_overlap, coarse_lo_method=coarse_lo_method,
    #               coarse_lo_certainty_threshold=args.coarse_lo_certainty_threshold,
    #               coarse_peak_blur_sigma=args.coarse_peak_blur_sigma, fine_width_partition=args.fine_width_partition,
    #               fine_height_partition=args.fine_height_partition, fine_overlap=args.fine_overlap,
    #               fine_lo_method=fine_lo_method, fine_lo_certainty_threshold=args.fine_lo_certainty_threshold,
    #               fine_peak_blur_sigma=args.fine_peak_blur_sigma, debug=args.debug)
    # run_method_over_dataset(args.output_dir, params = params)
    optimization()