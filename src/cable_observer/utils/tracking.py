from cable_observer.src.cable_observer.utils.utils import plot_paths
from image_processing import set_mask, process_image, preprocess_image, set_mask_3d
from paths_processing import get_gaps_length, get_linspaces, sort_paths, generate_paths, select_paths, \
    concatenate_paths, get_paths_and_gaps_length, inverse_path
from path import Path
import numpy as np
from time import time
from matplotlib import pyplot as plt


def track(frame, depth, last_spline_coords, params):
    t1 = time()
    # Get mask
    mask = frame if not params['input']['color'] else set_mask(frame, params['hsv'])
    # mask_depth = set_mask_3d(depth=depth, params_depth=params['depth'])

    if not np.any(mask):
        return False, [], [], mask, mask, None, None, [t1]
    # Preprocess image
    # mask = preprocess_image(img=mask_depth)
    mask = preprocess_image(img=mask)

    mask[depth > 1000] = 0  # filter out mask elements that are too far away
    mask[depth < 200] = 0  # filter out mask elements that are too close

    # Get image skeleton
    skeleton, paths_ends = process_image(img=mask)
    t2 = time()

    # Create paths
    paths = generate_paths(skeleton=skeleton, depth=depth, paths_ends=paths_ends, params_path=params['path'])
    t3 = time()

    # Get rid of too short paths
    paths = select_paths(paths=paths, params_path=params['path'])

    # plot_paths(paths)
    s = skeleton * 255
    d = depth / np.max(depth)
    dm = (d * mask).astype(np.uint8)
    ds = skeleton * d

    # Sort paths
    paths = sort_paths(paths=paths)
    t4 = time()
    #plt.subplot(121)
    #plt.imshow(d)
    #for p in paths:
    #    plt.subplot(121)
    #    plt.plot(p.coordinates[:, 1], p.coordinates[:, 0])
    #    plt.subplot(122)
    #    plt.plot(p.coordinates[:, 0], p.z_coordinates)
    #plt.show()

    # Calculate gaps between adjacent paths
    gaps = get_gaps_length(paths=paths)

    # Get a single linspace for a list of paths
    t = get_linspaces(paths=paths, gaps=gaps)

    # Concatenate all paths coordinates
    concatenated_paths_coords, concatenated_paths_z_coords = concatenate_paths(paths=paths)

    # Get spline representation for a concatenated paths
    full_length = get_paths_and_gaps_length(paths=paths, gaps=gaps)
    concatenated_paths = Path(coordinates=concatenated_paths_coords,
                              z_coordinates=concatenated_paths_z_coords,
                              length=full_length)

    # Inverse path if its needed to maintain direction to the previous iteration
    spline_coords, spline_params = inverse_path(path=concatenated_paths, last_spline_coords=last_spline_coords, t=t,
                                                between_grippers=params['path']['between_grippers'])
    t5 = time()

    # Find borders of the DLO and its width
    lower_bound, upper_bound = concatenated_paths.get_bounds(mask.astype(bool), spline_coords, common_width=True)

    return True, spline_coords, spline_params, skeleton.astype(np.float64) * 255, mask, lower_bound, upper_bound, \
           [t1, t2, t3, t4, t5]
