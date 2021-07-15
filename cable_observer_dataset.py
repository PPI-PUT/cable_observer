#!/usr/bin/env python3
from glob import glob
import numpy as np
import cv2
import argparse
from src.cable_observer.utils.debug_frame_processing import DebugFrameProcessing
from src.cable_observer.utils.tracking import track
from src.cable_observer.utils.image_processing import get_spline_image

parser = argparse.ArgumentParser(description='Cable observer using video input')
parser.add_argument('-d', '--debug', default=False, action='store_true', help="Debug mode")
parser.add_argument('-p', '--path', type=str, default="", help='Image file path')
args = parser.parse_args()


if __name__ == "__main__":
    # Debug params
    cps = []
    poc = []
    last_spline_coords = None
    for path in sorted(glob("/home/piotr/Downloads/vel_1.0_acc_1.0/wire/*.png")):
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
            cv2.imwrite(path.replace("wire", "spline"), 255 * img_spline_raw)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
