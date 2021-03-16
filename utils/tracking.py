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
    # Get image skeleton
    t1 = time()
    img, paths_ends = process_image(mask)
    t2 = time()
    #plt.imshow(img)
    #plt.show()

    # Create paths
    paths = []
    skel = np.pad(img, [[1, 1], [1, 1]], 'constant', constant_values=False)
    i = 0
    while len(paths_ends) > 0:
        coordinates, length = walk_faster(skel, tuple(paths_ends[0]))
        paths.append(Path(coordinates=coordinates, length=length))
        paths_ends.pop(0)
        paths_ends = remove_close_points((coordinates[-1][1], coordinates[-1][0]), paths_ends, max_px_gap=1)
        p = paths[-1]
        if length > 10:
            plt.plot(coordinates[:, 1], coordinates[:, 0], label=str(i))
            #plt.plot([p.begin[1], p.begin[1] + 10 * np.sin(p.begin_direction)], [p.begin[0], p.begin[0] + 10 * np.cos(p.begin_direction)], 'g')
            #plt.plot([p.end[1], p.end[1] + 10 * np.sin(p.end_direction)], [p.end[0], p.end[0] + 10 * np.cos(p.end_direction)], 'r')
            i+=1
    t3 = time()
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.legend()
    plt.show()


    # Get rid of too short paths
    paths = [p for p in paths if p.num_points > 3]
    paths = [p for p in paths if p.length > 20.]

    if len(paths) > 1:
        paths = sort_paths(paths)
    t4 = time()

    spline_params = []
    spline_coords = []
    for p in paths:
        # Calculate gaps between adjacent paths
        gaps_length = get_gaps_length(paths=p)

        # Get a single linespace for a list of paths
        t = get_linespaces(paths=p, gaps_length=gaps_length)

        # Merge all paths coordinates
        merged_paths = np.vstack([x() for x in p])
        # Get spline representation for a merged path
        full_length = np.sum([x.length for x in p]) + np.sum(gaps_length)
        merged_path = Path(coordinates=merged_paths, length=full_length)
        spline_coords.append(merged_path.get_spline(t=t))
        spline_params.append(merged_path.get_spline_params())
    t5 = time()

    # get bounds of a DLO
    #if lsc is not None:
    #    dist1 = np.sum(np.abs(lsc[0] - spline_coords[0]) + np.abs(lsc[-1] - spline_coords[-1]))
    #    dist2 = np.sum(np.abs(lsc[-1] - spline_coords[0]) + np.abs(lsc[0] - spline_coords[-1]))
    #    if dist2 < dist1:
    #        spline_coords = spline_coords[::-1]
    #        spline_params['coeffs'] = spline_params['coeffs'][:, ::-1]
    ##lower_bound, upper_bound = merged_path.get_bounds(mask, spline_coords, common_width=False)
    #lower_bound, upper_bound = merged_path.get_bounds(mask, spline_coords, common_width=True)
    lower_bound = None
    upper_bound = None

    return spline_coords, spline_params, img.astype(np.float64) * 255, mask, lower_bound, upper_bound, [t0, t1, t2, t3, t4, t5]
