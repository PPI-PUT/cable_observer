import cv2

from utils.image_processing import set_mask, process_image, preprocess_image
from utils.paths_processing import walk_fast, remove_close_points, get_gaps_length, get_linespaces, sort_paths, \
    walk_faster
from utils.path import Path
import numpy as np
from time import time
import matplotlib.pyplot as plt


def track(frame, lsc, masked=False):
    t0 = time()
    # Preprocess image
    mask = preprocess_image(frame, masked)
    #plt.imshow(mask)
    #plt.show()
    # Get image skeleton
    t1 = time()
    img, paths_ends = process_image(mask)
    t2 = time()

    # Create paths
    paths = []
    skel = np.pad(img, [[1, 1], [1, 1]], 'constant', constant_values=False)
    i = 0
    skip = 40
    while len(paths_ends) > 0:
        coordinates, length = walk_faster(skel, tuple(paths_ends[0]))
        paths.append(Path(coordinates=coordinates, length=length))
        paths_ends.pop(0)
        paths_ends = remove_close_points((coordinates[-1][1], coordinates[-1][0]), paths_ends, max_px_gap=1)
        p = paths[-1]
        if p.num_points > skip:
            plt.plot(coordinates[:, 1], coordinates[:, 0], label=str(i))
            plt.plot([p.begin[1], p.begin[1] + 10 * np.sin(p.begin_direction)], [p.begin[0], p.begin[0] + 10 * np.cos(p.begin_direction)], 'g')
            plt.plot([p.end[1], p.end[1] + 10 * np.sin(p.end_direction)], [p.end[0], p.end[0] + 10 * np.cos(p.end_direction)], 'r')
            i += 1
    t3 = time()
    plt.legend()
    plt.show()


    # Get rid of too short paths
    paths = [p for p in paths if p.num_points > skip]
    #paths = [p for p in paths if p.length > 30.]

    if len(paths) > 1:
        w, h = mask.shape
        diag = np.sqrt(w**2 + h**2)
        paths = sort_paths(paths, diag)
    t4 = time()

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
    t5 = time()

    # get bounds of a DLO
    if lsc is not None:
        dist1 = np.sum(np.abs(lsc[0] - spline_coords[0]) + np.abs(lsc[-1] - spline_coords[-1]))
        dist2 = np.sum(np.abs(lsc[-1] - spline_coords[0]) + np.abs(lsc[0] - spline_coords[-1]))
        if dist2 < dist1:
            spline_coords = spline_coords[::-1]
            spline_params['coeffs'] = spline_params['coeffs'][:, ::-1]
    #lower_bound, upper_bound = merged_path.get_bounds(mask, spline_coords, common_width=False)
    lower_bound, upper_bound = merged_path.get_bounds(mask, spline_coords, common_width=True)

    return spline_coords, spline_params, img.astype(np.float64) * 255, mask, lower_bound, upper_bound, [t0, t1, t2, t3, t4, t5]
