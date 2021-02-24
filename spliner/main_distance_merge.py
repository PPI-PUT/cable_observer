import cv2
import numpy as np
from time import time
from image_processing import set_mask, set_morphology, get_spline_image, find_ends, remove_if_more_than_3_neighbours
from paths_processing import walk_fast, remove_close_points, get_gaps_length, get_linespaces, sort_paths
from path import Path
import matplotlib.pyplot as plt

def main(frame, lsc):
    t0 = time()
    # Preprocess image
    img = set_mask(frame)
    mask = img

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
    if lsc is not None:
        dist1 = np.sum(np.abs(lsc[0] - spline_coords[0]) + np.abs(lsc[-1] - spline_coords[-1]))
        dist2 = np.sum(np.abs(lsc[-1] - spline_coords[0]) + np.abs(lsc[0] - spline_coords[-1]))
        if dist2 < dist1:
            spline_coords = spline_coords[::-1]
            spline_params['coeffs'] = spline_params['coeffs'][:, ::-1]
    lower_bound, upper_bound = merged_path.get_bounds(mask, spline_coords)

    print("INNER T1", t1 - t0)
    print("INNER T2", t2 - t1)
    print("INNER T3", t3 - t2)
    print("INNER T4", t4 - t3)
    print("INNER T5", t5 - t4)
    print("INNER T6", t6 - t5)

    return spline_coords, spline_params, img.astype(np.float64) * 255, mask, lower_bound, upper_bound


if __name__ == "__main__":
    #cap = cv2.VideoCapture(2)
    cap = cv2.VideoCapture("./output_v4_short.avi")
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Skip blank frames
    for i in range(100):
        _, frame = cap.read()

    cps = []
    poc = []
    w = 0
    last_spline_coords = None
    plot = True
    while True:
        t0 = time()
        _, frame = cap.read()

        # Get spline coordinates
        t1 = time()
        spline_coords, spline_params, img_skeleton, mask, lower_bound, upper_bound = main(frame, last_spline_coords)
        last_spline_coords = spline_coords
        t2 = time()

        img_spline = get_spline_image(spline_coords=spline_coords, shape=frame.shape)
        t3 = time()
        if plot:
            img_low = get_spline_image(spline_coords=lower_bound, shape=frame.shape)
            img_up = get_spline_image(spline_coords=upper_bound, shape=frame.shape)
            img_spline = np.stack([img_low[:, :, 0], img_spline[:, :, 1], img_up[:, :, 2]], axis=-1)

            idx = np.where(np.any(img_spline, axis=-1))
            #frame[idx[0], idx[1], 1] = (255*img_spline[idx[0], idx[1], 1]).astype(np.uint8)
            frame[idx[0], idx[1]] = (255*img_spline[idx[0], idx[1]]).astype(np.uint8)

            z = np.zeros_like(mask)
            mask = np.stack([mask, z, z], axis=-1)
            mask[idx[0], idx[1], 1] = img_spline[idx[0], idx[1], 1]

            z = np.zeros_like(img_skeleton)
            img_skeleton = np.stack([img_skeleton, z, z], axis=-1)
            img_skeleton[idx[0], idx[1], 1] = img_spline[idx[0], idx[1], 1]

            coeffs = spline_params['coeffs'].astype(np.int32)
            for i in range(-2, 3):
                for j in range(-2, 3):
                    frame[coeffs[0] + i, coeffs[1] + j, :] = np.array([0, 0, 255], dtype=np.uint8)
            cps.append(coeffs)
            k = 25
            d = int(spline_coords.shape[0] / k) + 1
            poc.append(spline_coords[::d])
        if w > 300:
            break
        w += 1

        # Show outputs
        cv2.imshow('spline', img_spline)
        cv2.imshow('skel', img_skeleton)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        t4 = time()
        print("T1", t1 - t0)
        print("T2", t2 - t1)
        print("T3", t3 - t2)
        print("T4", t4 - t3)
        print("T ALL", t4 - t0)

    if plot:
        cps = np.stack(cps, axis=0)
        #for i in range(2):
        for i in range(cps.shape[-1]):
            plt.plot(cps[:, 0, i], cps[:, 1, i], label=str(i))
        plt.legend()
        plt.show()

        poc = np.stack(poc, axis=0)
        for i in range(poc.shape[1]):
            plt.plot(poc[:, i, 0], poc[:, i, 1], label=str(i))
        plt.legend()
        plt.show()


    cap.release()
    cv2.destroyAllWindows()
