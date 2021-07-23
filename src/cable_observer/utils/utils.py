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


def create_spline_dir(dir_path):
    """
    Create a directory for splines.
    :param dir_path: path to directory of image
    :type dir_path: str
    """
    subdataset_path = os.path.dirname(dir_path.rstrip('/')) # get the parent dir
    spline_path = os.path.join(subdataset_path, "spline")
    Path(spline_path).mkdir(parents=True, exist_ok=True)
