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
    lower_bound, upper_bound = concatenated_paths.get_bounds(mask, spline_coords, common_width=True)

    return spline_coords, spline_params, skeleton.astype(np.float64) * 255, mask, lower_bound, upper_bound, [t1, t2, t3, t4, t5]
