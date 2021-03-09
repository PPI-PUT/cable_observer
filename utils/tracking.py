from utils.image_processing import set_mask, set_morphology, find_ends, remove_if_more_than_3_neighbours
from utils.paths_processing import walk_fast, remove_close_points, get_gaps_length, get_linespaces, sort_paths, \
    walk_faster
from utils.path import Path
import numpy as np


def track(frame, lsc, masked=False):
    # Preprocess image
    img = frame if masked else set_mask(frame)

    # Get image skeleton
    img, mask = set_morphology(img)

    img = remove_if_more_than_3_neighbours(img)

    # Find paths ends
    paths_ends = find_ends(img)

    # Create paths
    paths = []
    skel = np.pad(img, [[1, 1], [1, 1]], 'constant', constant_values=False)
    while len(paths_ends) > 0:
        coordinates, length = walk_faster(skel, tuple(paths_ends[0]))
        paths.append(Path(coordinates=coordinates, length=length))
        paths_ends.pop(0)
        paths_ends = remove_close_points((coordinates[-1][1], coordinates[-1][0]), paths_ends, max_px_gap=1)


    # Get rid of too short paths
    paths = [p for p in paths if p.num_points > 3]
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
    full_length = np.sum([p.length for p in paths]) + np.sum(gaps_length)
    merged_path = Path(coordinates=merged_paths, length=full_length)
    spline_coords = merged_path.get_spline(t=t)
    spline_params = merged_path.get_spline_params()

    # get bounds of a DLO
    if lsc is not None:
        dist1 = np.sum(np.abs(lsc[0] - spline_coords[0]) + np.abs(lsc[-1] - spline_coords[-1]))
        dist2 = np.sum(np.abs(lsc[-1] - spline_coords[0]) + np.abs(lsc[0] - spline_coords[-1]))
        if dist2 < dist1:
            spline_coords = spline_coords[::-1]
            spline_params['coeffs'] = spline_params['coeffs'][:, ::-1]
    #lower_bound, upper_bound = merged_path.get_bounds(mask, spline_coords, common_width=False)
    lower_bound, upper_bound = merged_path.get_bounds(mask, spline_coords, common_width=True)

    return spline_coords, spline_params, img.astype(np.float64) * 255, mask, lower_bound, upper_bound
