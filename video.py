from time import time

import cv2
import numpy as np
from utils.image_processing import get_spline_image
from utils.tracking import track

if __name__ == "__main__":
    #cap = cv2.VideoCapture(2)
    #cap = cv2.VideoCapture("videos/output.avi")
    #cap = cv2.VideoCapture("videos/output_v4_short.avi")
    cap = cv2.VideoCapture("videos/output_v4.avi")
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Skip blank frames
    for i in range(100):
        _, frame = cap.read()

    cps = []
    poc = []
    ious = []
    last_spline_coords = None
    debug = True
    while True:
        _, frame = cap.read()
        #frame = cv2.resize(frame, (1280, 960))

        t0 = time()
        # Get spline coordinates
        spline_coords, spline_params, img_skeleton, mask, lower_bound, upper_bound, t = track(frame, last_spline_coords)
        t1 = time()
        print("TIME", t1 - t0)

        img_spline = get_spline_image(spline_coords=spline_coords, shape=frame.shape)
        last_spline_coords = spline_coords

        if debug:
            ts = 0
            for i in range(1, len(t)):
                print("T", i, t[i] - t[i-1])
                ts += (t[i] - t[i-1])
            print("TIME OURS", ts)

            pred = np.zeros_like(frame)[..., 0]
            N = 10
            d = 0
            for i in range(N+1+2*d):
                t = (i - d) / N
                coords = lower_bound * t + upper_bound * (1 - t)
                uv = np.around(coords).astype(np.int32)
                pred[np.clip(uv[:, 0], 0, pred.shape[0] - 1), np.clip(uv[:, 1], 0, pred.shape[1] - 1)] = 255

            pred = cv2.dilate(pred, np.ones((3, 3)))
            pred = cv2.erode(pred, np.ones((3, 3)))
            intersection = (mask * pred) > 0
            union = (mask + pred) > 0
            iou = np.sum(intersection.astype(np.float32)) / np.sum(union.astype(np.float32))
            ious.append(iou)
            #imu = (np.logical_xor(union, intersection)).astype(np.float32)
            imu = mask.astype(np.float32) - pred.astype(np.float32)
            umi = union.astype(np.float32) - intersection.astype(np.float32)
            print(iou)

            img_low = get_spline_image(spline_coords=lower_bound, shape=frame.shape)
            img_up = get_spline_image(spline_coords=upper_bound, shape=frame.shape)
            img_spline = np.stack([img_low[:, :, 0], img_spline[:, :, 1], img_up[:, :, 2]], axis=-1)

            idx = np.where(np.any(img_spline, axis=-1))
            for i in range(-2, 3):
                for j in range(-2, 3):
                    frame[idx[0] + i, idx[1] + j] = (255*img_spline[idx[0], idx[1]]).astype(np.uint8)
            #frame[idx[0], idx[1]] = (255*img_spline[idx[0], idx[1]]).astype(np.uint8)

            z = np.zeros_like(mask)
            mask = np.stack([mask, z, z], axis=-1)
            mask[idx[0], idx[1], 1] = img_spline[idx[0], idx[1], 1]

            z = np.zeros_like(img_skeleton)
            img_skeleton = np.stack([img_skeleton, z, z], axis=-1)
            img_skeleton[idx[0], idx[1], 1] = img_spline[idx[0], idx[1], 1]

            coeffs = spline_params['coeffs'].astype(np.int32)
            for i in range(-2, 3):
                for j in range(-2, 3):
                    frame[np.clip(coeffs[0] + i, 0, frame.shape[0] - 1), np.clip(coeffs[1] + j, 0, frame.shape[1] - 1), :] =\
                        np.array([0, 0, 255], dtype=np.uint8)
            cps.append(coeffs)
            k = 25
            d = int(spline_coords.shape[0] / k) + 1
            poc.append(spline_coords[::d])
            print("MEAN:", np.mean(ious))

            cv2.imshow('mask', mask)
            cv2.imshow('pred', pred)
            cv2.imshow('imu', imu)
            cv2.imshow('umi', umi)
            cv2.imshow('spline', img_spline)
            cv2.imshow('skel', img_skeleton)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
