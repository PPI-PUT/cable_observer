import os
from pathlib import Path


def get_spline_path(img_path):
    """
    Get a new path for splines.
    :param img_path: path to image including image filename
    :type img_path: str
    :return: new spline path including image filename
    :rtype: str
    """
    img_name = img_path.split("/")[-1]
    subdataset_path = img_path.split("/")[:-2]
    spline_path = ["spline", img_name]
    return "/".join(subdataset_path + spline_path)


def create_spline_dir(kwargs, default_path):
    """
    Create a directory for splines.
    :param kwargs: input kwargs
    :type kwargs: list
    :param default_path: default directory for splines
    :type default_path: str
    """
    # Find any path
    dir_path = [d[1] for d in kwargs if isinstance(d[1], str) and len(d[1]) > 0]

    # Check if path exist, otherwise set default path
    dir_path = dir_path[0] if dir_path else default_path

    subdataset_path = os.path.dirname(dir_path.rstrip('/'))  # get the parent dir
    spline_path = os.path.join(subdataset_path, "spline")
    Path(spline_path).mkdir(parents=True, exist_ok=True)
    return spline_path
