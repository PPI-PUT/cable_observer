import cv2
import numpy as np

from image_processing import set_mask, set_morphology, get_spline_image
from paths_processing import walk_fast, find_ends, remove_close_points, keys_to_sequence, reorder_paths,\
    get_gaps_length, get_linespaces
from path import Path
import matplotlib.pyplot as plt

def init(frame):
    """
    Get initial spline.
    :param frame: camera input frame (W x H x 3)
    :type frame: np.array
    :return: x & y coordinates as a buffer, skeleton image
    :rtype: np.array, np.array
    """
    # Preprocess image
    #img = set_mask(frame)

    # Get image skeleton
    img = set_morphology(frame[..., 0])

    # Find path ends
    path_ends = find_ends(img=img, max_px_gap=5)
    assert len(path_ends) <= 2, cv2.imwrite("init_path_error.png", img.astype(np.float64) * 255) and \
                                "More than 2 path ends detected. Wrong initialization input. " \
                                "Path saved as init_path_error.png. Path ends: " + str(path_ends)

    # Get coordinates sequence
    coordinates, length = walk_fast(img, tuple(path_ends[0]))
    path = Path(coordinates=coordinates, length=length)

    # Get spline representation
    t = np.linspace(0., 1., path.num_points)
    x, y = path.get_spline(t=t)

    buffer = np.column_stack((x, y))
    return buffer, img.astype(np.float64) * 255


def main(frame, buffer):
    # Preprocess image
    #img = set_mask(frame)

    # Get image skeleton
    img = set_morphology(frame[..., 0])

    # Find paths ends
    paths_ends = find_ends(img=img, max_px_gap=5)

    # Create paths
    paths = []
    while len(paths_ends) > 0:
        coordinates, length = walk_fast(img, tuple(paths_ends[0]))
        paths.append(Path(coordinates=coordinates, length=length))
        paths_ends.pop(0)
        paths_ends = remove_close_points((coordinates[-1][1], coordinates[-1][0]), paths_ends)
        x = [-p[0] for p in coordinates]
        y = [p[1] for p in coordinates]
        plt.plot(y, x)
    plt.show()
    for path in paths:
        x = [-p[0] for p in path.coordinates]
        y = [p[1] for p in path.coordinates]
        plt.plot(y, x)
    plt.show()

    # Get rid of too short paths
    paths = [p for p in paths if p.num_points > 1]

    # Compare buffer with paths and get paths sequence
    paths_keys = []
    for path in paths:
        key = path.get_key(buffer=buffer)
        paths_keys.append(key)
        x = [-p[0] for p in path.coordinates]
        y = [p[1] for p in path.coordinates]
        plt.plot(y, x)
    plt.show()


    # Get paths sequence based on keys
    sequence = keys_to_sequence(keys=paths_keys)

    # Reorder paths using new sequence
    paths = reorder_paths(paths=paths, sequence=sequence)

    # Calculate gaps between adjacent paths
    gaps_length = get_gaps_length(paths=paths)

    # Get a single linespace for a list of paths
    t = get_linespaces(paths=paths, gaps_length=gaps_length)

    # Merge all paths coordinates
    merged_paths = np.vstack([p() for p in paths])

    # Get spline representation for a merged path
    merged_path = Path(coordinates=merged_paths, length=-1)
    x, y = merged_path.get_spline(t=t)

    return x, y, img.astype(np.float64) * 255


if __name__ == "__main__":
    init_frame = cv2.imread("../a_init.png")
    test_frame = cv2.imread("../a_test.png")
    buffer, init_img_skeleton = init(init_frame)

        # Get spline coordinates
    x, y, test_img_skeleton = main(test_frame, buffer)

    buffer = np.column_stack((x, y))

        # Convert spline coordinates to image frame
    img_spline = get_spline_image(x=x, y=y, shape=test_frame.shape)

    plt.subplot(231)
    plt.imshow(init_frame)
    plt.subplot(232)
    plt.imshow(init_img_skeleton)
    plt.subplot(233)
    plt.imshow(test_frame)
    plt.subplot(234)
    plt.imshow(test_img_skeleton)
    plt.subplot(235)
    plt.imshow(img_spline)
    plt.show()
