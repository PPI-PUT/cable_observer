import cv2
from utils.image_processing import get_spline_image
import matplotlib.pyplot as plt
from utils.tracking_figure import track

if __name__ == "__main__":
    test_frame = cv2.imread("imgs/023.png")[..., 0]
    plt.imshow(test_frame, cmap='gray')
    plt.axis('off')
    d = 221 / 385 * 5
    plt.gcf().set_size_inches(d, d)
    plt.savefig("fig1.png", bbox_inches='tight', pad_inches=0)

    # Get spline coordinates
    spline_coords, spline_params, img_skeleton, mask, lower_bound, upper_bound = track(test_frame, None, masked=True)

    # Convert spline coordinates to image frame
    img_spline = get_spline_image(spline_coords, test_frame.shape)

    control_points = spline_params['coeffs']

    #plt.subplot(331)
    #plt.imshow(test_frame, cmap='gray')
    #plt.subplot(332)
    #plt.imshow(img_skeleton, cmap='gray')
    #plt.subplot(333)
    #plt.plot(control_points[1], control_points[0], 'rx')
    #plt.imshow(img_spline)
    #plt.subplot(334)
    #plt.imshow(test_frame - img_skeleton, cmap='gray')
    #plt.show()

    plt.plot(control_points[1], control_points[0], 'rx', markersize=3)
    plt.imshow(img_spline, cmap='gray')
    plt.axis('off')
    plt.gcf().set_size_inches(d, d)
    plt.savefig("fig5.png", bbox_inches='tight', pad_inches=0)

