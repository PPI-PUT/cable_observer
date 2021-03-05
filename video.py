import cv2
import numpy as np
from utils.image_processing import get_spline_image
from utils.tracking import track

if __name__ == "__main__":
    #cap = cv2.VideoCapture(2)
    #cap = cv2.VideoCapture("videos/output.avi")
    cap = cv2.VideoCapture("videos/output_v4.avi")
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Skip blank frames
    for i in range(100):
        _, frame = cap.read()

    cps = []
    poc = []
    last_spline_coords = None
    plot = True
    while True:
        _, frame = cap.read()

        # Get spline coordinates
        spline_coords, spline_params, img_skeleton, mask, lower_bound, upper_bound = track(frame, last_spline_coords)
        last_spline_coords = spline_coords

        img_spline = get_spline_image(spline_coords=spline_coords, shape=frame.shape)
        if plot:
            img_low = get_spline_image(spline_coords=lower_bound, shape=frame.shape)
            img_up = get_spline_image(spline_coords=upper_bound, shape=frame.shape)
            img_spline = np.stack([img_low[:, :, 0], img_spline[:, :, 1], img_up[:, :, 2]], axis=-1)

            idx = np.where(np.any(img_spline, axis=-1))
            frame[idx[0], idx[1]] = (255*img_spline[idx[0], idx[1]]).astype(np.uint8)

            z = np.zeros_like(mask)
            mask = np.stack([mask, z, z], axis=-1)
            mask[idx[0], idx[1], 1] = img_spline[idx[0], idx[1], 1]

            z = np.zeros_like(img_skeleton)
            img_skeleton = np.stack([img_skeleton, z, z], axis=-1)
            img_skeleton[idx[0], idx[1], 1] = img_spline[idx[0], idx[1], 1]

            coeffs = spline_params['coeffs'].astype(np.int32)
            for i in range(-2, 3):
                for j in range(-2, 3):
                    frame[coeffs[0] + i, coeffs[1] + j, :] = np.array([0, 0, 255], dtype=np.uint8)
            cps.append(coeffs)
            k = 25
            d = int(spline_coords.shape[0] / k) + 1
            poc.append(spline_coords[::d])

        # Show outputs
        cv2.imshow('spline', img_spline)
        cv2.imshow('skel', img_skeleton)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
