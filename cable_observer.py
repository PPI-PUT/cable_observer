#!/usr/bin/env python3
import cv2
import argparse
from src.cable_observer.utils.debug_frame_processing import DebugFrameProcessing
from src.cable_observer.utils.tracking import track
from src.cable_observer.utils.image_processing import get_spline_image

parser = argparse.ArgumentParser(description='Cable observer using video input')
parser.add_argument('-d', '--debug', default=False, action='store_true', help="Debug mode")
parser.add_argument('-c', '--camera', default=False, action='store_true', help="Use camera input")
parser.add_argument('-i', '--input', type=int, default="-1", help='Camera input number')
parser.add_argument('-p', '--path', type=str, default="./videos/output_v4.avi", help='Video file path')
parser.add_argument('-b', '--between', default=False, action='store_true', help='Set to true if you want to receive a '
                                                                                'spline between horizontally oriented'
                                                                                ' grippers')
args = parser.parse_args()


if __name__ == "__main__":
    if args.camera:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.path)

    # Debug params
    last_spline_coords = None
    dfp = DebugFrameProcessing()

    # Skip blank frames
    for i in range(100):
        _, frame = cap.read()

    while True:
        _, frame = cap.read()
        spline_coords, spline_params, skeleton, mask,  lower_bound, upper_bound, t = track(frame, last_spline_coords,
                                                                                           between_grippers=args.between)
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
            cv2.imshow('frame', frame)
            cv2.imshow('spline', img_spline_raw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
