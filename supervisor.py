import cv2 
import matplotlib.pyplot as plt
from time import perf_counter
import argparse
from mask_segmentation import MaskSegmentation
from path_processing import PathProcessing
from spline_mask import SplineMask

parser = argparse.ArgumentParser(description='Cable observer')
parser.add_argument('-f', '--filepath', help='Path to image', required=False, type=str, default="./imgs/005.png")
parser.add_argument('-k', '--knots', help='Spline knots', required=False, type=int, default=35)
parser.add_argument('-l', '--min-path-length', help='Minimum length of path to consider', required=False, type=int, default=10)
parser.add_argument('-p', '--min-path-points', help='Minimum points of path to consider', required=False, type=int, default=3)
parser.add_argument('--hsv-mask', help='HSV masking [h_min, h_max, s_min, s_max, v_min, v_max]',
                    required=False, type=list, default=[175, 8, 130, 255, 0, 255])
parser.add_argument('-m', '--is-masked', help='Already masked input (binary image)',
                    required=False, type=bool, default=True)
parser.add_argument('-d', '--max-dilations', help='Maximum num of image dilations',
                    required=False, type=int, default=50)
parser.add_argument('-v', '--dir-vec-length', help='Direction vector length (points)',
                    required=False, type=int, default=10)
parser.add_argument('--output-dilation', help='Num of output mask dilation',
                    required=False, type=int, default=3)
args = vars(parser.parse_args())

class CableObserver():
    def __init__(self, is_masked, hsv_params, knots, min_path_length, min_path_points, max_dilations, dir_vec_length,
                 output_dilation):
        self.max_dilations = max_dilations
        self.output_dilation = output_dilation
        self.segmentation = MaskSegmentation(is_masked=is_masked, hsv_params=hsv_params)
        self.paths = PathProcessing(knots=knots, min_path_length=min_path_length, min_path_points=min_path_points,
                                    dir_vec_length=dir_vec_length)
        self.spline_mask = SplineMask()

    def exec(self, input_image):
        iterations = 0
        t1 = perf_counter()
        while (self.paths.success == False):
            # mask segmentation
            mask_image = self.segmentation.exec(input_image, dilate_it=iterations)
            # paths processing
            spline_coords, spline_params = self.paths.exec(mask_image=mask_image, paths_ends=self.segmentation.paths_ends)
            iterations += 1
            if iterations > self.max_dilations:
                raise RuntimeError("Reached maximum input mask dilation.")

        t2 = perf_counter()

        # math repr (spline) to image
        spline_mask = self.spline_mask.exec(splines=spline_coords, shape=input_image.shape, output_dilation=self.output_dilation)
        t3 = perf_counter()
        print(f"{(t2-t1)*1000}, {(t3-t2)*1000} -> {(t3-t1)*1000} [ms]")

        return spline_mask, spline_coords
    

if __name__ == "__main__":
    p_filepath = args["filepath"]
    p_hsv_params = {"hue": {"min": args["hsv_mask"][0], "max": args["hsv_mask"][1]},
                  "saturation": {"min": args["hsv_mask"][2], "max": args["hsv_mask"][3]},
                  "value": {"min": args["hsv_mask"][4], "max": args["hsv_mask"][5]}}
    p_knots = args["knots"]
    p_min_path_length = args["min_path_length"]
    p_min_path_points = args["min_path_points"]
    p_is_masked = args["is_masked"]
    p_max_dilations = args["max_dilations"]
    p_dir_vec_length = args["dir_vec_length"]
    p_output_dilation = args["output_dilation"]

    img = cv2.imread(p_filepath, cv2.IMREAD_COLOR)
    p = CableObserver(is_masked=p_is_masked, hsv_params=p_hsv_params, knots=p_knots, min_path_length=p_min_path_length,
                      min_path_points=p_min_path_points, max_dilations=p_max_dilations, dir_vec_length=p_dir_vec_length,
                      output_dilation=p_output_dilation)
    output_mask, spline_coords = p.exec(input_image=img)
    print(f"Num of splines: {len(spline_coords)}")
    plt.imshow(output_mask)
    plt.axis("off")
    plt.tight_layout(pad=0.1)
    plt.show()
