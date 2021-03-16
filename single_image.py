from glob import glob

import cv2
from utils.image_processing import get_spline_image
import matplotlib.pyplot as plt
from utils.tracking import track
import numpy as np

if __name__ == "__main__":
    #test_frame = cv2.imread("imgs/024.png")[..., 0]
    for f in glob("imgs/025.jpg"):
    #for f in glob("../cables_dataset/cable_dataset_simple/*_mask_all.png"):
        test_frame = cv2.imread(f)[..., 0]

        # Get spline coordinates
        spline_coords, spline_params, img_skeleton, mask, lower_bound, upper_bound, t = track(test_frame, None, masked=True)

        c = ['r', 'g', 'b', 'c', 'm', 'tan', 'olive', 'indigo', 'lavender', 'lime', 'royalblue', 'springgreen']
        resultant_img = np.zeros_like(test_frame).astype(np.float64)
        for i in range(len(spline_params)):
        # Convert spline coordinates to image frame
            img_spline = get_spline_image(spline_coords[i], test_frame.shape)

            control_points = spline_params[i]['coeffs']

            plt.plot(control_points[1], control_points[0], 'x', color=c[i % len(c)], markersize=3)
            plt.axis('off')
            #plt.gcf().set_size_inches(d, d)
            resultant_img += img_spline
        plt.imshow(resultant_img, cmap='gray')
        #plt.show()
        #plt.savefig(f.replace("mask_all", "paths"), bbox_inches='tight', pad_inches=0)
        plt.clf()

