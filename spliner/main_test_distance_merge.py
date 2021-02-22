import cv2
import numpy as np

from image_processing import set_mask, set_morphology, get_spline_image, remove_if_more_than_3_neighbours, find_ends
from paths_processing import walk_fast, remove_close_points, get_gaps_length, get_linespaces, sort_paths
from path import Path
import matplotlib.pyplot as plt


def main(frame):
    # Preprocess image
    #img = set_mask(frame)

    # Get image skeleton
    skel = set_morphology(frame[..., 0])
    img = frame[..., 0]

    plt.subplot(121)
    plt.imshow(skel)
    skel = remove_if_more_than_3_neighbours(skel)
    plt.subplot(122)
    plt.imshow(skel)
    #plt.show()
    plt.clf()

    # Find paths ends
    #paths_ends = find_ends(img=skel, max_px_gap=1)
    paths_ends = find_ends(skel)
    for i, pe in enumerate(paths_ends):
        plt.plot(pe[0], -pe[1], 'rx', label=str(i))
    plt.legend()
    #plt.show()

    # Create paths
    paths = []
    while len(paths_ends) > 0:
        coordinates, length = walk_fast(skel, tuple(paths_ends[0]))
        #coordinates, length = walk(img, img, tuple(paths_ends[0]), 5, 3)
        paths.append(Path(coordinates=coordinates, length=length))
        paths_ends.pop(0)
        for coo in coordinates:
            paths_ends = remove_close_points((coo[1], coo[0]), paths_ends, max_px_gap=1)
        #paths_ends = remove_close_points((coordinates[-1][1], coordinates[-1][0]), paths_ends)
        x = [-p[0] for p in coordinates]
        y = [p[1] for p in coordinates]
        plt.plot(y, x)
        #plt.show()
        pass
    plt.show()

    # Get rid of too short paths
    paths = [p for p in paths if p.num_points > 1]
    paths = [p for p in paths if p.length > 10.]

    if len(paths) > 1:
        paths = sort_paths(paths)

    # Calculate gaps between adjacent paths
    gaps_length = get_gaps_length(paths=paths)

    # Get a single linespace for a list of paths
    t = get_linespaces(paths=paths, gaps_length=gaps_length)

    # Merge all paths coordinates
    merged_paths = np.vstack([p() for p in paths])

    # Get spline representation for a merged path
    merged_path = Path(coordinates=merged_paths, length=-1)
    spline_coords = merged_path.get_spline(t=t)
    spline_params = merged_path.get_spline_params()

    return spline_coords, spline_params, img.astype(np.float64) * 255


if __name__ == "__main__":
    #test_frame = cv2.imread("../a_test.png")
    test_frame = cv2.imread("../a_test_2.png")
    #test_frame = cv2.imread("../c_test.png")
    #test_frame = cv2.imread("../test_v3.png")
    #test_frame = cv2.imread("../u_hard.png")
    #test_frame = cv2.imread("../si_1.png")
    #buffer, init_img_skeleton = init(init_frame)

        # Get spline coordinates
    spline_coords, spline_params, test_img_skeleton = main(test_frame)

        # Convert spline coordinates to image frame
    img_spline = get_spline_image(spline_coords, test_frame.shape)

    control_points = spline_params['coeffs']

    plt.subplot(221)
    plt.imshow(test_frame)
    plt.subplot(222)
    plt.imshow(test_img_skeleton)
    plt.subplot(223)
    plt.imshow(img_spline)
    plt.plot(control_points[1], control_points[0], 'rx')
    plt.show()

