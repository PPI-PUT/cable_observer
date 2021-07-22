import os
from glob import glob
import numpy as np
import pandas as pd
import cv2
import argparse
from src.cable_observer.utils.debug_frame_processing import DebugFrameProcessing
from src.cable_observer.utils.tracking import track
from src.cable_observer.utils.image_processing import get_spline_image
from src.cable_observer.utils.utils import get_spline_path, create_spline_dir

parser = argparse.ArgumentParser(description='Cable observer processes the cable masks into spline control points'
                                             'and points along the path and saves it into spline.csv in the dataset folder')
parser.add_argument('-d', '--debug', default=False, action='store_true', help="Debug mode")
parser.add_argument('-p', '--path', type=str, default="", help='Dataset path')
parser.add_argument('-s', '--save', default=False, action='store_true', help='If true then saves also the images of the splines')
parser.add_argument('--knots', type=int, default=25, help='Number of knots in the estimated spline')
parser.add_argument('--pts', type=int, default=256, help='Number of points along the estimated spline')
parser.add_argument('-b', '--between', default=False, action='store_true', help='Set to true if you want to receive a '
                                                                                'spline between horizontally oriented'
                                                                                ' grippers')
args = parser.parse_args()

#args.path = "/remodel_ws/src/wire_manipulations_dataset/media/remodel_dataset/wire/0.003/left_diagonal_right_circular/vel_1.0_acc_1.0/wire"
args.path = "/home/piotr/Downloads/vel_1.0_acc_1.0/wire"


if __name__ == "__main__":
    # Debug params
    df = pd.DataFrame()
    last_spline_coords = None
    create_spline_dir(dir_path=args.path)
    dfp = DebugFrameProcessing()
    for i, path in enumerate(sorted(glob(os.path.join(args.path, "*.png"))[:10])):
        print(path)
        frame = cv2.imread(path)
        frame = (np.sum(frame, axis=-1) > 0).astype(np.float32)
        spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t = track(frame, last_spline_coords,
                                                                                          masked=True,
                                                                                          between_grippers=args.between,
                                                                                          num_of_knots=args.knots,
                                                                                          num_of_pts=args.pts)
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if args.save:
            img_spline_raw = get_spline_image(spline_coords=spline_coords, shape=frame.shape)
            cv2.imwrite(get_spline_path(img_path=path), 255 * img_spline_raw)
        sample = pd.DataFrame({"image_filename": os.path.basename(path),
                               "control_points_x": [spline_params['coeffs'][0], ],
                               "control_points_y": [spline_params['coeffs'][1], ],
                               "points_on_curve_x": [spline_coords[:, 0], ],
                               "points_on_curve_y": [spline_coords[:, 1], ],
                               }, index=[0])
        sample.index = [i]
        df = df.append(sample)
    csv_path = os.path.join("/".join(args.path.split("/")[:-1]), "spline.csv")
    df.to_csv(csv_path)



