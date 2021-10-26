from .image_processing import set_mask, process_image, preprocess_image
from .paths_processing import get_gaps_length, get_linspaces, sort_paths, generate_paths, select_paths, \
    concatenate_paths, get_paths_and_gaps_length, inverse_path
from .path import Path
import numpy as np
from time import time


def track(frame, last_spline_coords, params):
    t1 = time()
    # Get mask
    mask = frame if not params['input']['color'] else set_mask(frame, params['hsv'])

    # Preprocess image
    mask = preprocess_image(mask)

    # Get image skeleton
    skeleton, paths_ends = process_image(mask)
    t2 = time()

    # Create paths
    paths = generate_paths(skeleton=skeleton, paths_ends=paths_ends, params_path=params['path'])
    t3 = time()

    # Get rid of too short paths
    paths = select_paths(paths)

    # Sort paths
    paths = sort_paths(paths)
    t4 = time()

    # Calculate gaps between adjacent paths
    gaps = get_gaps_length(paths=paths)

    # Get a single linspace for a list of paths
    t = get_linspaces(paths=paths, gaps=gaps)

    # Concatenate all paths coordinates
    concatenated_paths_coords = concatenate_paths(paths=paths)

    # Get spline representation for a concatenated paths
    full_length = get_paths_and_gaps_length(paths=paths, gaps=gaps)
    concatenated_paths = Path(coordinates=concatenated_paths_coords, length=full_length)

    # Inverse path if its needed to maintain direction to the previous iteration
    spline_coords, spline_params = inverse_path(path=concatenated_paths, last_spline_coords=last_spline_coords, t=t,
                                                between_grippers=params['path']['between_grippers'])
    t5 = time()

    # Find borders of the DLO and its width
    lower_bound, upper_bound = concatenated_paths.get_bounds(mask.astype(bool), spline_coords, common_width=True)

    ########## Get dense spline points ##########
    dense_T = np.linspace(0., 1., 2000)
    dense_spline_coords = np.column_stack((concatenated_paths.y_spline(dense_T), concatenated_paths.x_spline(dense_T)))
    img_spline = np.zeros(shape=(frame.shape[0], frame.shape[1]), dtype=np.uint8)
    dense_points = np.unique(np.round(dense_spline_coords).astype(int), axis=0)
    dense_x_points = np.clip(dense_points[:, 0], 0, frame.shape[1] - 1)
    dense_y_points = np.clip(dense_points[:, 1], 0, frame.shape[0] - 1)
    img_spline[dense_y_points, dense_x_points] = 255
    ###############################################

    return spline_coords, spline_params, skeleton.astype(np.float64) * 255, mask, lower_bound, upper_bound, [t1, t2, t3, t4, t5], img_spline
