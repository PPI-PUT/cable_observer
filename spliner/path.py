import numpy as np
from scipy.interpolate import LSQUnivariateSpline


class Path:
    def __init__(self, coordinates, length):
        self.coordinates = np.array(coordinates).reshape((-1, 2))
        self.length = length
        self.num_points = len(coordinates)
        self.begin = self.coordinates[0]
        self.end = self.coordinates[-1]

    def __call__(self):
        return self.coordinates

    def flip_path(self):
        """
        Reverse sequence of coordinates.
        """
        self.coordinates = np.flip(self.coordinates, axis=0)
        self.begin = self.coordinates[0]
        self.end = self.coordinates[-1]

    def get_spline(self, t):
        """
        Get spline coordinates.
        :param t: path linespace
        :type t: np.array
        :return: x coordinates, y coordinates, x spline, y spline
        :rtype: np.array, np.array, np.array, np.array
        """
        xys = np.stack(self.coordinates, axis=0)
        T = np.linspace(0., 1., 128)
        k = 7
        knots = np.linspace(0., 1., k)[1:-1]
        x_spline = LSQUnivariateSpline(t, xys[:, 0], knots)
        y_spline = LSQUnivariateSpline(t, xys[:, 1], knots)
        x = x_spline(T)
        y = y_spline(T)
        return x, y, x_spline, y_spline

    def get_key(self, buffer):
        """
        Get closest key of previous path spline representation.
        :param buffer: previous spline
        :type buffer: np.array
        :return: closest key
        :rtype: int
        """
        distances_begin = [np.linalg.norm(buffer[k] - self.begin) for k in np.arange(0, len(buffer))]
        distances_end = [np.linalg.norm(buffer[k] - self.end) for k in np.arange(0, len(buffer))]

        key_begin = distances_begin.index(min(distances_begin))
        key_end = distances_end.index(min(distances_end))

        key = min(key_begin, key_end)

        if key_end < key_begin:
            self.flip_path()

        return key