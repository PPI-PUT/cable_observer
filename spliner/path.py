import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt


class Path:
    def __init__(self, coordinates, length):
        self.coordinates = np.array(coordinates).reshape((-1, 2))
        self.length = length
        self.x_spline = None
        self.y_spline = None
        self.num_points = len(coordinates)
        self.begin = self.coordinates[0]
        self.end = self.coordinates[-1]
        self.T = np.linspace(0., 1., 128)
        if self.length > 10:
            bv = self.coordinates[0] - self.coordinates[5]
            ev = self.coordinates[-1] - self.coordinates[-6]
            self.begin_direction = np.arctan2(bv[0], bv[1])
            self.end_direction = np.arctan2(ev[0], ev[1])

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
        :param t: path linspace
        :type t: np.array
        :return: x coordinates, y coordinates, x spline, y spline
        :rtype: np.array, np.array, np.array, np.array
        """
        xys = np.stack(self.coordinates, axis=0)
        k = 25 - 4
        d = int((t.shape[0] - 2) / k) + 1
        knots = t[1:-1:d]
        self.x_spline = LSQUnivariateSpline(t, xys[:, 0], knots)
        self.y_spline = LSQUnivariateSpline(t, xys[:, 1], knots)
        spline_coords = np.column_stack((self.x_spline(self.T), self.y_spline(self.T)))
        return spline_coords

    def get_spline_params(self):
        """
        Get spline parameters:
            * length - full length of spline including gaps
            * coeffs - spline coefficients [x, y]
            * residuals - spline residuals [x, y]
            * derivatives - spline derivatives [x, y]
        :return: spline params
        :rtype: dict
        """
        coeffs = np.array([self.x_spline.get_coeffs(), self.y_spline.get_coeffs()])
        residuals = np.array([self.x_spline.get_residual(), self.y_spline.get_residual()])
        cps = np.linspace(0, 1, 7)

        derivatives = np.array([[self.x_spline.derivatives([d]) for d in cps[1:-1]],
                                [self.y_spline.derivatives([d]) for d in cps[1:-1]]])

        return {"length": self.length, "coeffs": coeffs, "residuals": residuals, "derivatives": derivatives}

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

    def get_bounds(self, mask, spline_coords):
        x = spline_coords[:, 0]
        y = spline_coords[:, 1]
        normal_emp = np.diff(spline_coords, axis=0)
        normal_emp = np.stack([-normal_emp[:, 1], normal_emp[:, 0]], axis=-1)
        normal_emp = normal_emp / np.linalg.norm(normal_emp, axis=-1, keepdims=True)
        normal_emp = np.concatenate([normal_emp, normal_emp[-1:]], axis=0)
        n = 40
        div = 4.
        x_normal_seq = np.around(
            x[:, np.newaxis] + n / div * normal_emp[:, :1] * (np.linspace(-1., 1., 2 * n)[np.newaxis])).astype(
            np.int32)
        y_normal_seq = np.around(
            y[:, np.newaxis] + n / div * normal_emp[:, -1:] * (np.linspace(-1., 1., 2 * n)[np.newaxis])).astype(np.int32)
        r = mask[x_normal_seq, y_normal_seq]
        width = np.sum(r, axis=-1)
        #mean_width = np.mean(width) / div
        mean_width = np.median(width) / div
        print("WIDTH:", mean_width)
        lower_bound = spline_coords + normal_emp * mean_width / 2
        upper_bound = spline_coords - normal_emp * mean_width / 2

        return lower_bound, upper_bound
