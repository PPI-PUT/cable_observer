import numpy as np
from scipy.interpolate import LSQUnivariateSpline


import matplotlib.pyplot as plt
class Path:
    def __init__(self, coordinates, z_coordinates, length, num_of_knots=25, num_of_pts=256, max_width=40, width_step=4,
                 vector_dir_len=5):
        self.coordinates = np.array(coordinates).reshape((-1, 2))
        self.z_coordinates = z_coordinates
        self.length = length
        self.x_spline = None
        self.y_spline = None
        self.z_spline = None
        self.num_points = len(coordinates)
        self.begin = self.coordinates[0]
        self.end = self.coordinates[-1]
        self.T = np.linspace(0., 1., num_of_pts)
        self.k = num_of_knots
        self.max_width = max_width
        self.width_step = width_step
        begin_vector = self.coordinates[0] - self.coordinates[min(vector_dir_len, len(self.coordinates) - 1)]
        end_vector = self.coordinates[-1] - self.coordinates[max(-vector_dir_len - 1, -len(self.coordinates))]
        self.begin_direction = np.arctan2(begin_vector[0], begin_vector[1])
        self.end_direction = np.arctan2(end_vector[0], end_vector[1])

    def __call__(self):
        return self.coordinates

    def flip_path(self):
        """
        Reverse sequence of coordinates.
        """
        self.coordinates = np.flip(self.coordinates, axis=0)
        self.z_coordinates = np.flip(self.z_coordinates, axis=0)
        self.begin = self.coordinates[0]
        self.end = self.coordinates[-1]

    def get_spline(self, t, between_grippers=False):
        """
        Get spline coordinates.
        :param t: path linspace
        :type t: np.array
        :param between_grippers: boolean which decides if take care only about the cable
                                 between horizontally oriented grippers
                                 (extracts the part of a cable between extreme spline extrema)
        :type between_grippers: bool
        :return: x coordinates, y coordinates, x spline, y spline
        :rtype: np.array, np.array, np.array, np.array
        """
        xys = np.stack(self.coordinates, axis=0)
        zs = self.z_coordinates
        k = self.k - 4
        #d_ = int((t.shape[0] - 2) / k) + 1
        d = np.linspace(1., t.shape[0] - 2, k).astype(np.int32)
        knots = t[d]
        #knots_ = t[1:-1:d_]
        self.x_spline = LSQUnivariateSpline(t, xys[:, 0], knots)
        self.y_spline = LSQUnivariateSpline(t, xys[:, 1], knots)
        valid = zs != 0
        t_v = t[valid]
        #knots_v = t_v[1:-1:d]
        d = np.linspace(1., t_v.shape[0] - 2, k).astype(np.int32)
        knots_v = t_v[d]
        z_v = zs[valid]
        self.z_spline = LSQUnivariateSpline(t_v, z_v, knots_v)

        knots_ = np.linspace(0., 1., self.k - 2)[1:-1]
        self.x_spline = LSQUnivariateSpline(self.T, self.x_spline(self.T), knots_)
        self.y_spline = LSQUnivariateSpline(self.T, self.y_spline(self.T), knots_)
        self.z_spline = LSQUnivariateSpline(self.T, self.z_spline(self.T), knots_)
        spline_coords = np.column_stack((self.x_spline(self.T), self.y_spline(self.T), self.z_spline(self.T)))
        #plt.subplot(221)
        #plt.plot(spline_coords[:, 0], spline_coords[:, 1])
        #plt.subplot(222)
        #plt.plot(np.linspace(0., 1., z_v.shape[0]), z_v, 'rx')
        #plt.plot(self.T, spline_coords[:, -1])
        #plt.show()

        # quickfix to ignore part of the cable which is not between the grippers
        if between_grippers:
            xs = self.x_spline(self.T)
            ys = self.y_spline(self.T)
            zs = self.z_spline(self.T)
            dx = xs[1:] - xs[:-1]
            dy = ys[1:] - ys[:-1]
            ratio = dx / dy
            rm = np.abs(ratio) < 0.1
            idxs = np.where(rm)[0]
            xs = xs[idxs[0]:idxs[-1]]
            ys = ys[idxs[0]:idxs[-1]]
            t = np.linspace(0., 1., idxs[-1] - idxs[0])
            d = int((t.shape[0] - 2) / k) + 1
            knots = t[1:-1:d]
            self.x_spline = LSQUnivariateSpline(t, xs, knots)
            self.y_spline = LSQUnivariateSpline(t, ys, knots)
            self.z_spline = LSQUnivariateSpline(t, zs, knots)
            spline_coords = np.column_stack((self.x_spline(self.T), self.y_spline(self.T), self.z_spline(self.T)))
        return spline_coords

    def get_spline_params(self):
        """
        Get spline parameters:
            * length - full length of spline including gaps
            * coeffs - spline coefficients [x, y, z]
            * residuals - spline residuals [x, y, z]
        :return: spline params
        :rtype: dict
        """
        coeffs = np.array([self.x_spline.get_coeffs(), self.y_spline.get_coeffs(), self.z_spline.get_coeffs()])
        residuals = np.array([self.x_spline.get_residual(), self.y_spline.get_residual(), self.z_spline.get_residual()])

        return {"length": self.length, "coeffs": coeffs, "residuals": residuals}

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

    def get_bounds(self, mask, spline_coords, common_width=True):
        """
        Find borders of the DLO and its width
        :param mask: image of the DLO mask
        :type mask: np.array
        :param spline_coords: coordinates of the points on the spline fitted to DLO
        :type spline_coords: np.array
        :return: lower and upper bound spline coordinates of the DLO
        :rtype: np.array, np.array
        """
        spline_coords = spline_coords[:, :2]
        mask = np.pad(mask, [[self.max_width, self.max_width], [self.max_width, self.max_width]], 'constant', constant_values=False)
        x = spline_coords[:, 0] + self.max_width
        y = spline_coords[:, 1] + self.max_width
        normal_emp = np.diff(spline_coords, axis=0)
        normal_emp = np.stack([-normal_emp[:, 1], normal_emp[:, 0]], axis=-1)
        normal_emp = normal_emp / np.linalg.norm(normal_emp, axis=-1, keepdims=True)
        normal_emp = np.concatenate([normal_emp, normal_emp[-1:]], axis=0)
        x_normal_seq = np.around(x[:, np.newaxis] +
                                 self.max_width / self.width_step * normal_emp[:, :1]
                                 * (np.linspace(-1., 1., 2 * self.max_width)[np.newaxis])).astype(np.int32)
        y_normal_seq = np.around(y[:, np.newaxis] +
                                 self.max_width / self.width_step * normal_emp[:, -1:]
                                 * (np.linspace(-1., 1., 2 * self.max_width)[np.newaxis])).astype(np.int32)
        r = mask[np.clip(x_normal_seq, 0, mask.shape[0] - 1), np.clip(y_normal_seq, 0, mask.shape[1] - 1)]
        width = np.sum(r, axis=-1) / self.width_step
        if common_width:
            mean_width = np.median(width)
            ds = normal_emp * mean_width / 2 - 0.5
            lower_bound = spline_coords + ds
            upper_bound = spline_coords - ds
            k = self.k - 4
            d = int((self.T.shape[0] - 2) / k) + 1
            knots = self.T[1:-1:d]
            x_spline = LSQUnivariateSpline(self.T, upper_bound[:, 0], knots)
            y_spline = LSQUnivariateSpline(self.T, upper_bound[:, 1], knots)
            upper_bound = np.column_stack((x_spline(self.T), y_spline(self.T)))
            x_spline = LSQUnivariateSpline(self.T, lower_bound[:, 0], knots)
            y_spline = LSQUnivariateSpline(self.T, lower_bound[:, 1], knots)
            lower_bound = np.column_stack((x_spline(self.T), y_spline(self.T)))
        else:
            lower_bound = spline_coords + normal_emp * width[:, np.newaxis] / 2
            upper_bound = spline_coords - normal_emp * width[:, np.newaxis] / 2

        return lower_bound, upper_bound
