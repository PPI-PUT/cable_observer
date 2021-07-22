#!/usr/bin/env python3
from glob import glob
import numpy as np
import cv2
from src.cable_observer.utils.debug_frame_processing import DebugFrameProcessing
from src.cable_observer.utils.tracking import track
from src.cable_observer.utils.image_processing import get_spline_image

if __name__ == "__main__":
    # Debug params
    last_spline_coords = None
    dfp = DebugFrameProcessing()
    #for path in ["./ds/extracted_wire/wire_000013.png"]:
    #for path in ["./wire_000013.png"]:
    for path in ["/home/piotr/Downloads/vel_1.0_acc_1.0/wire/000001.png"]:
        frame = cv2.imread(path)
        mask = (np.sum(frame, axis=-1) > 0).astype(np.float32)
        spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t = track(mask, last_spline_coords, True)

        dfp.set_debug_state(frame, last_spline_coords, spline_coords, spline_params, skeleton, mask, lower_bound,
                            upper_bound, t)
        dfp.run_debug_sequence()
        dfp.print_t()
        dbg = "imgs/debug/"
        cv2.imwrite(dbg + 'frame.png', dfp.img_frame)
        cv2.imwrite(dbg + 'mask.png', dfp.img_mask)
        cv2.imwrite(dbg + 'prediction.png', dfp.img_pred)
        cv2.imwrite(dbg + 'spline.png', dfp.img_spline)
        cv2.imwrite(dbg + 'skeleton.png', dfp.img_skeleton)
