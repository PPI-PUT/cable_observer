import cv2
import numpy as np

from image_processing import set_mask, set_morphology, get_spline_image
from paths_processing import walk_fast, find_ends, remove_close_points, keys_to_sequence, reorder_paths,\
    get_gaps_length, get_linespaces
from path import Path

def init(frame):
    """
    Get initial spline.
    :param frame: camera input frame (W x H x 3)
    :type frame: np.array
    :return: x & y coordinates as a buffer, skeleton image
    :rtype: np.array, np.array
    """
    # Preprocess image
    img = set_mask(frame)

    # Get image skeleton
    img = set_morphology(img)

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
    x, y, x_spline, y_spline = path.get_spline(t=t)

    buffer = np.column_stack((x, y))
    coeffs = np.array([x_spline.get_coeffs(), y_spline.get_coeffs()])

    return buffer, coeffs, img.astype(np.float64) * 255


def main(frame, buffer):
    # Preprocess image
    img = set_mask(frame)

    # Get image skeleton
    img = set_morphology(img)

    # Find paths ends
    paths_ends = find_ends(img=img, max_px_gap=5)

    # Create paths
    paths = []
    while len(paths_ends) > 0:
        coordinates, length = walk_fast(img, tuple(paths_ends[0]))
        paths.append(Path(coordinates=coordinates, length=length))
        paths_ends.pop(0)
        paths_ends = remove_close_points((coordinates[-1][1], coordinates[-1][0]), paths_ends)

    # Get rid of too short paths
    paths = [p for p in paths if p.num_points > 3]

    # Compare buffer with paths and get paths sequence
    paths_keys = []
    for path in paths:
        key = path.get_key(buffer=buffer)
        paths_keys.append(key)

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
    x, y, x_spline, y_spline = merged_path.get_spline(t=t)
    coeffs = np.array([x_spline.get_coeffs(), y_spline.get_coeffs()])

    return x, y, coeffs, img.astype(np.float64) * 255


if __name__ == "__main__":
    cap = cv2.VideoCapture(6)

    # Skip blank frames
    for i in range(100):
        _, frame = cap.read()

    # Initialization spline
    _, frame = cap.read()
    buffer, coeffs_buffer, img_skeleton = init(frame)

    while True:
        _, frame = cap.read()

        # Get spline coordinates
        x, y, coeffs, img_skeleton = main(frame, buffer)
        new_spline = np.column_stack((x, y))

        # Calculate error
        #dst = np.linalg.norm(new_spline[:, np.newaxis] - buffer[np.newaxis], axis=-1)
        #err = np.sum(np.min(dst, axis=0))
        #err = np.sum(np.fabs(new_spline - buffer))
        err = np.sum(np.absolute(coeffs - coeffs_buffer))
        print(err)

        # Check error
        if err < 1000:
            # Write buffer variable for next loop
            buffer = new_spline
            # Convert spline coordinates to image frame
            img_spline = get_spline_image(x=x, y=y, shape=frame.shape)
        else:
            # Keep previous spline
            img_spline = get_spline_image(x=buffer[..., 0], y=buffer[..., 1], shape=frame.shape)

        # Show outputs
        cv2.imshow('spline', img_spline)
        cv2.imshow('frame', img_skeleton)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
