import os
from glob import glob
from time import time
import pandas as pd
import cv2
import argparse
import yaml
from src.cable_observer.utils.debug_frame_processing import DebugFrameProcessing
from src.cable_observer.utils.tracking import track
from src.cable_observer.utils.image_processing import get_spline_image
from src.cable_observer.utils.utils import get_spline_path, create_spline_dir
import numpy as np


parser = argparse.ArgumentParser(description='Cable observer processes the cable masks into spline control points'
                                             'and points along the path and saves it into spline.csv in the dataset folder')
parser.add_argument('-i', '--images', type=str, default="", help='Images path')
parser.add_argument('-d', '--debug', default=False, action='store_true', help="Debug output")
parser.add_argument('-s', '--save_dataframe', default=False, action='store_true', help='If true then saves splines metadata')
parser.add_argument('-o', '--save_output', default=False, action='store_true', help='If true then saves the images of the splines')
args = parser.parse_args()

# Config
stream = open(os.path.dirname(__file__) + "/config/params.yaml", 'r')
params = yaml.load(stream, Loader=yaml.FullLoader)

# Spline output file
spline_path = create_spline_dir(kwargs=args._get_kwargs(), default_path=os.path.abspath(__file__))
df = pd.DataFrame()

# Runtime vars
last_spline_coords = None
dfp = DebugFrameProcessing()


def main(frame, depth, img_spline_path, dataframe_index):
    flag_shutdown = False
    spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t = track(frame=frame,
                                                                                      depth=depth,
                                                                                      last_spline_coords=last_spline_coords,
                                                                                      params=params)
    if args.debug:
        dfp.set_debug_state(frame, last_spline_coords, spline_coords, spline_params, skeleton, mask, lower_bound,
                            upper_bound, t)
        dfp.run_debug_sequence()
        dfp.print_t()
        cv2.imshow('frame', dfp.img_frame)
        cv2.imshow('mask', dfp.img_mask)
        cv2.imshow('prediction', dfp.img_pred)
        cv2.imshow('spline', dfp.img_spline)
        cv2.imshow('skeleton', dfp.img_skeleton)
    else:
        img_spline_raw = get_spline_image(spline_coords=spline_coords, shape=frame.shape)
        cv2.imshow('spline', img_spline_raw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        flag_shutdown = True

    if args.save_output:
        img_spline_raw = get_spline_image(spline_coords=spline_coords, shape=frame.shape)
        cv2.imwrite(get_spline_path(img_path=img_spline_path), 255 * img_spline_raw)

    # Generate dataframe sample for current spline
    spline_metadata = pd.DataFrame({"image_filename": os.path.basename(img_spline_path),
                           "control_points_x": [spline_params['coeffs'][1], ],
                           "control_points_y": [spline_params['coeffs'][0], ],
                           "control_points_z": [spline_params['coeffs'][2], ],
                           "points_on_curve_x": [spline_coords[:, 1], ],
                           "points_on_curve_y": [spline_coords[:, 0], ],
                           "points_on_curve_z": [spline_coords[:, 2], ],
                           }, index=[0])
    spline_metadata.index = [dataframe_index]

    return spline_metadata, flag_shutdown


if __name__ == "__main__":
    # Dataset input
    if args.images:
        for i, (path_rgb, path_depth) in enumerate(zip(sorted(glob(os.path.join(args.images, "rgb_*.png"))),
                                                       sorted(glob(os.path.join(args.images, "depth_*.txt"))))):
            #if i < 325:
            #    continue
            print(path_rgb)
            frame = cv2.imread(path_rgb, params['input']['color'])  # cv2.IMREAD_GRAYSCALE / cv2.IMREAD_COLOR flag
            depth = np.loadtxt(path_depth)  # cv2.IMREAD_GRAYSCALE / cv2.IMREAD_COLOR flag
            spline_metadata, flag_shutdown = main(frame=frame, depth=depth, img_spline_path=path_rgb, dataframe_index=i)
            df = df.append(spline_metadata)

    if args.save_dataframe:
        csv_path = os.path.join("/".join(spline_path.split("/")[:-1]), "spline.csv")
        df.to_csv(csv_path)

    cv2.destroyAllWindows()
