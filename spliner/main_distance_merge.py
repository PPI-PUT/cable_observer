import cv2
import numpy as np
from time import time
from image_processing import set_mask, set_morphology, get_spline_image, find_ends, remove_if_more_than_3_neighbours
from paths_processing import walk_fast, remove_close_points, get_gaps_length, get_linespaces, sort_paths
from path import Path
import matplotlib.pyplot as plt

def main(frame):
    t0 = time()
    # Preprocess image
    img = set_mask(frame)

    t1 = time()
    # Get image skeleton
    img = set_morphology(img)

    img = remove_if_more_than_3_neighbours(img)

    t2 = time()
    # Find paths ends
    #paths_ends = find_ends(img=img, max_px_gap=5)
    paths_ends = find_ends(img)

    t3 = time()
    # Create paths
    paths = []
    while len(paths_ends) > 0:
        coordinates, length = walk_fast(img, tuple(paths_ends[0]))
        paths.append(Path(coordinates=coordinates, length=length))
        paths_ends.pop(0)
        paths_ends = remove_close_points((coordinates[-1][1], coordinates[-1][0]), paths_ends, max_px_gap=1)

    # Get rid of too short paths
    paths = [p for p in paths if p.num_points > 3]
    paths = [p for p in paths if p.length > 10.]
    t4 = time()

    if len(paths) > 1:
        paths = sort_paths(paths)

    t5 = time()
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
    t6 = time()

    print("INNER T1", t1 - t0)
    print("INNER T2", t2 - t1)
    print("INNER T3", t3 - t2)
    print("INNER T4", t4 - t3)
    print("INNER T5", t5 - t4)
    print("INNER T6", t6 - t5)

    return spline_coords, spline_params, img.astype(np.float64) * 255


if __name__ == "__main__":
    cap = cv2.VideoCapture(2)
    #cap = cv2.VideoCapture("~/Videos/kabel.avi")
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Skip blank frames
    for i in range(100):
        _, frame = cap.read()

    while True:
        t0 = time()
        _, frame = cap.read()

        # Get spline coordinates
        t1 = time()
        spline_coords, spline_params, img_skeleton = main(frame)
        t2 = time()

        img_spline = get_spline_image(spline_coords=spline_coords, shape=frame.shape)
        t3 = time()

        # Show outputs
        cv2.imshow('spline', img_spline)
        cv2.imshow('frame', img_skeleton)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        t4 = time()
        print("T1", t1 - t0)
        print("T2", t2 - t1)
        print("T3", t3 - t2)
        print("T4", t4 - t3)
        print("T ALL", t4 - t0)

    cap.release()
    cv2.destroyAllWindows()
