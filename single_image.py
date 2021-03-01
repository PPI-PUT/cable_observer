import cv2
from utils.image_processing import get_spline_image
import matplotlib.pyplot as plt
from utils.tracking import track

if __name__ == "__main__":
    test_frame = cv2.imread("imgs/011.png")[..., 0]

    # Get spline coordinates
    spline_coords, spline_params, img_skeleton, mask, lower_bound, upper_bound = track(test_frame, None, masked=True)

    # Convert spline coordinates to image frame
    img_spline = get_spline_image(spline_coords, test_frame.shape)

    control_points = spline_params['coeffs']

    plt.subplot(221)
    plt.imshow(test_frame)
    plt.subplot(222)
    plt.imshow(img_skeleton)
    plt.subplot(223)
    plt.imshow(img_spline)
    plt.plot(control_points[1], control_points[0], 'rx')
    plt.show()

