import autograd.numpy as np
from autograd import grad
from autograd.numpy import log, sqrt, sin, cos, exp, pi, prod
from autograd.numpy.random import normal, uniform


class Flower:
    def __init__(self, radius=1, petals=5, move_out=3):
        self.radius = radius
        self.petals = petals
        self.move_out = move_out

    def position(self, t):
        t = t % 1

        amplitude = np.sin(2 * self.petals * np.pi * t) + self.move_out
        x = amplitude * np.cos(2 * np.pi * t)
        y = amplitude * np.sin(2 * np.pi * t)
        return np.array([x, y]) / 3.5 * self.radius

    def unit_normal(self, t):
        t = t % 1

        amplitude = np.sin(2 * self.petals * np.pi * t) + self.move_out
        delta_amplitude = 2 * self.petals * np.pi * np.sin(2 * self.petals * np.pi * t)
        cos_2pi_t = np.cos(2 * np.pi * t)
        sin_2pi_t = np.sin(2 * np.pi * t)
        tangent_x = delta_amplitude * cos_2pi_t - sin_2pi_t * 2 * np.pi * amplitude
        tangent_y = delta_amplitude * sin_2pi_t + cos_2pi_t * 2 * np.pi * amplitude

        normal_vector = np.array([-tangent_y, tangent_x])
        return np.array([normal_vector]) / np.sqrt(np.sum(normal_vector ** 2))


class Heart:
    def __init__(self, radius=1):
        self.radius = radius

    def position(self, t):
        t = t % 1
        x = 16 * np.sin(2 * np.pi * t) ** 3
        y = 13 * np.cos(2 * np.pi * t) - 5 * np.cos(4 * np.pi * t) - 2 * np.cos(6 * np.pi * t) - np.cos(8 * np.pi * t)
        return np.array([x, y]) / 10 * self.radius

    def unit_normal(self, t):
        t = t % 1
        # ignore scale since we only care about unit vector
        tangent_x = 96 * np.pi * np.sin(2 * np.pi * t) ** 2 * np.cos(2 * np.pi * t)
        tangent_y = -26 * np.pi * np.sin(2 * np.pi * t) + 20 * np.pi * np.sin(4 * np.pi * t) \
                    + 12 * np.pi * np.sin(6 * np.pi * t) + 8 * np.pi * np.sin(8 * np.pi * t)

        normal_vector = np.array([-tangent_y, tangent_x])
        return np.array([normal_vector]) / np.sqrt(np.sum(normal_vector ** 2))


class Polygon:
    def __init__(self, corners=8, radius=1.):
        self.corners = corners
        self.edges = np.empty((0, 2))
        self.radius = radius

        for i in range(corners):
            pos = i / corners
            point = np.array([[np.sin(2 * np.pi * pos), np.cos(2 * np.pi * pos)]]) * radius
            self.edges = np.concatenate((self.edges, point), axis=0)

        # include the first point again to yield a circle
        self.edges = np.concatenate((self.edges, np.array([[0, 1]]) * radius), axis=0)

    def position(self, t):  # input needs to be numpy array
        t = t % 1 * self.corners
        t_remain, t_idx = t % 1, (t // 1).astype(int)

        point = (1 - t_remain).reshape(-1, 1) * self.edges[t_idx,] \
                + t_remain.reshape(-1, 1) * self.edges[t_idx + 1,]
        return point.transpose()

    def unit_normal(self, t):
        t = t % 1 * self.corners
        t_remain, t_idx = t % 1, (t // 1).astype(int)
        tangent_vector = self.edges[t_idx + 1,] - self.edges[t_idx,]
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        return np.array([normal_vector]) / np.sqrt(np.sum(normal_vector ** 2))


class Cross:
    def __init__(self, radius=1, grid=0.05):
        self.edges = np.empty((0, 2))

        C = 2 * np.pi
        x1, x2, x3, x4 = np.sin(C * (0.75 - grid)), np.sin(C * (-grid)), np.sin(C * (grid)), np.sin(C * (0.25 - grid))
        y1, y2, y3, y4 = np.sin(C * (0.25 - grid)), np.sin(C * (grid)), np.sin(C * (-grid)), np.sin(C * (0.75 - grid))

        self.edges = np.concatenate((self.edges, np.array([[x3, y1]])), axis=0)
        self.edges = np.concatenate((self.edges, np.array([[x3, y2]])), axis=0)
        self.edges = np.concatenate((self.edges, np.array([[x4, y2]])), axis=0)

        self.edges = np.concatenate((self.edges, np.array([[x4, y3]])), axis=0)
        self.edges = np.concatenate((self.edges, np.array([[x3, y3]])), axis=0)
        self.edges = np.concatenate((self.edges, np.array([[x3, y4]])), axis=0)

        self.edges = np.concatenate((self.edges, np.array([[x2, y4]])), axis=0)
        self.edges = np.concatenate((self.edges, np.array([[x2, y3]])), axis=0)
        self.edges = np.concatenate((self.edges, np.array([[x1, y3]])), axis=0)

        self.edges = np.concatenate((self.edges, np.array([[x1, y2]])), axis=0)
        self.edges = np.concatenate((self.edges, np.array([[x2, y2]])), axis=0)
        self.edges = np.concatenate((self.edges, np.array([[x2, y1]])), axis=0)

        # include the first point again to yield a circle
        self.edges = np.concatenate((self.edges, np.array([[x3, y1]])), axis=0)

        self.edges = self.edges * radius

    def position(self, t):
        t = t % 1 * 12
        t_remain, t_idx = t % 1, (t // 1).astype(int)

        point = (1 - t_remain).reshape(-1, 1) * self.edges[t_idx,] \
                + t_remain.reshape(-1, 1) * self.edges[t_idx + 1,]
        return point.transpose()

    def unit_normal(self, t):
        t = t % 1 * 12
        t_remain, t_idx = t % 1, (t // 1).astype(int)
        tangent_vector = self.edges[t_idx + 1,] - self.edges[t_idx,]
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        return np.array([normal_vector]) / np.sqrt(np.sum(normal_vector ** 2))


class Star:
    def __init__(self, radius=1, rotate=0.05, theta=0.618):
        C = 2 * np.pi

        points = []
        for i in np.array([0, 0.2, 0.4, 0.6, 0.8, 0, 0.2]) + rotate:
            points.append([np.cos(C * i), np.sin(C * i)])

        self.edges = np.empty((0, 2))
        for idx in [0, 1, 2, 3, 4, 0]:
            self.edges = np.concatenate((self.edges, np.array([points[idx]])), axis=0)
            inner_point = np.array([points[idx]]) * theta + np.array([points[idx + 2]]) * (1 - theta)
            self.edges = np.concatenate((self.edges, inner_point), axis=0)

        self.edges = self.edges * radius

    def position(self, t):
        t = t % 1 * 10
        t_remain, t_idx = t % 1, (t // 1).astype(int)

        point = (1 - t_remain).reshape(-1, 1) * self.edges[t_idx,] \
                + t_remain.reshape(-1, 1) * self.edges[t_idx + 1,]
        return point.transpose()

    def unit_normal(self, t):
        t = t % 1 * 10
        t_remain, t_idx = t % 1, (t // 1).astype(int)
        tangent_vector = self.edges[t_idx + 1,] - self.edges[t_idx,]
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        return np.array([normal_vector]) / np.sqrt(np.sum(normal_vector ** 2))



