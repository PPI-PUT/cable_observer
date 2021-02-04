import cv2
import numpy as np
from time import time
from image_processing import set_mask, set_morphology, get_spline_image
from paths_processing import walk_fast, find_ends, remove_close_points, keys_to_sequence, reorder_paths,\
    get_gaps_length, get_linespaces, get_errors
from path import Path


def init(frame):
    """
    Get initial spline.
    :param frame: camera input frame (W x H x 3)
    :type frame: np.array
    :return: x & y coordinates as a buffer, path coefficients, skeleton image
    :rtype: np.array, np.array, np.array
    """
    # Preprocess image
    img = set_mask(frame)

    # Get image skeleton
    img = set_morphology(img)

    # Find path ends
    paths_ends, _ = find_ends(img=img, max_px_gap=5)
    assert len(paths_ends) <= 2, cv2.imwrite("init_path_error.png", img.astype(np.float64) * 255) and \
                                "More than 2 paths ends detected. Wrong initialization input. " \
                                "Image saved as init_path_error.png. Path ends: " + str(paths_ends)
    assert len(paths_ends) > 0, cv2.imwrite("init_path_error.png", img.astype(np.float64) * 255) and \
                                "0 paths ends detected. Wrong initialization input. " \
                                "Image saved as init_path_error.png. Path ends: " + str(paths_ends)

    # Get coordinates sequence
    coordinates, length = walk_fast(img, tuple(paths_ends[0]))
    path = Path(coordinates=coordinates, length=length)

    # Get spline representation
    t = np.linspace(0., 1., path.num_points)
    spline_coords = path.get_spline(t=t)
    spline_params = path.get_spline_params()
    spline_length = path.length

    return spline_coords, spline_params, spline_length, img.astype(np.float64) * 255


def main(frame, buffer):
    # Preprocess image
    img = set_mask(frame)

    # Get image skeleton
    img = set_morphology(img)

    # Find paths ends and remove branches connection
    paths_ends, img = find_ends(img=img, max_px_gap=5)

    # Create paths
    paths = []
    while len(paths_ends) > 0:
        coordinates, length = walk_fast(img, tuple(paths_ends[0]))
        paths.append(Path(coordinates=coordinates, length=length))
        paths_ends.pop(0)
        paths_ends = remove_close_points((coordinates[-1][1], coordinates[-1][0]), paths_ends)

    # Get rid of too short paths
    paths = [p for p in paths if p.num_points > 10]

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
    full_length = np.sum([p.length for p in paths]) + np.sum(gaps_length)
    merged_path = Path(coordinates=merged_paths, length=full_length)
    spline_coords = merged_path.get_spline(t=t)
    spline_params = merged_path.get_spline_params()
    spline_length = merged_path.length

    return spline_coords, spline_params, spline_length, img.astype(np.float64) * 255


if __name__ == "__main__":
    cap = cv2.VideoCapture(6)

    # Skip blank frames
    for i in range(100):
        _, frame = cap.read()

    # Initialization spline
    _, frame = cap.read()
    spline_coords_buffer, spline_params_buffer, spline_length_buffer, img_skeleton = init(frame)

    while True:
        _, frame = cap.read()

        # Get spline coordinates
        spline_coords, spline_params, spline_length, img_skeleton = main(frame, spline_coords_buffer)

        # Calculate error
        # dst = np.linalg.norm(spline_coords[:, np.newaxis] - spline_coords_buffer[np.newaxis], axis=-1)
        # err = np.sum(np.min(dst, axis=0))
        # err = np.sum(np.fabs(new_spline - buffer))
        # err = np.sum(np.absolute(spline_params['coeffs'] - spline_params_buffer['coeffs']))
        errors = get_errors(spline_params, spline_params_buffer)
        err = np.max(errors['coeffs_error_max'])

        length_diff = spline_length - spline_length_buffer
        print("Diff length: ", length_diff)

        # Check error
        if np.fabs(length_diff) < 100:
            # Write buffer variable for next loop
            spline_coords_buffer = spline_coords
            spline_params_buffer = spline_params
            spline_length_buffer = spline_length
            # Convert spline coordinates to image frame
            img_spline = get_spline_image(spline_coords=spline_coords, shape=frame.shape)
        else:
            # Keep previous spline
            img_spline = get_spline_image(spline_coords=spline_coords_buffer, shape=frame.shape)

        # Show outputs
        cv2.imshow('spline', img_spline)
        cv2.imshow('frame', img_skeleton)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
