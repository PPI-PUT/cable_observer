import os
from glob import glob
import numpy as np
import cv2
import argparse
from src.cable_observer.utils.debug_frame_processing import DebugFrameProcessing
from src.cable_observer.utils.tracking import track
from src.cable_observer.utils.image_processing import get_spline_image
from src.cable_observer.utils.utils import get_spline_path, create_spline_dir

parser = argparse.ArgumentParser(description='Cable observer using video input')
parser.add_argument('-d', '--debug', default=False, action='store_true', help="Debug mode")
parser.add_argument('-p', '--path', type=str, default="", help='Dataset path')
args = parser.parse_args()

args.path = "/remodel_ws/src/wire_manipulations_dataset/media/remodel_dataset/wire/0.003/left_diagonal_right_circular/vel_1.0_acc_1.0/wire"


if __name__ == "__main__":
    # Debug params
    cps = []
    poc = []
    last_spline_coords = None
    create_spline_dir(dir_path=args.path)
    for path in sorted(glob(os.path.join(args.path, "*.png"))):
        print(path)
    #for path in glob("./ds/extracted_wire/wire*"):
        frame = cv2.imread(path)
        frame = (np.sum(frame, axis=-1) > 0).astype(np.float32)
        spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t = track(frame, last_spline_coords, True)
        if args.debug:
            dfp = DebugFrameProcessing(frame, cps, poc, last_spline_coords,
                                      spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t)
            cps, poc, last_spline_coords = dfp.get_params()
            dfp.print_t()
            cv2.imshow('frame', dfp.img_frame)
            cv2.imshow('mask', dfp.img_mask)
            cv2.imshow('prediction', dfp.img_pred)
            cv2.imshow('spline', dfp.img_spline)
            cv2.imshow('skeleton', dfp.img_skeleton)
        else:
            img_spline_raw = get_spline_image(spline_coords=spline_coords, shape=frame.shape)
            #cv2.imshow('frame', frame)
            #cv2.imwrite(path.replace("wire_", "spline_"), 255 * img_spline_raw)
            cv2.imwrite(get_spline_path(img_path=path), 255 * img_spline_raw)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
