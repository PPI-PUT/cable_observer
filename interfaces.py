from abc import ABC, abstractmethod
from numpy import ndarray

class Segmentation(ABC):
    '''
    Base class for the implementation of segmentation methods.
    '''

    @abstractmethod
    def __init__(self, **kwargs):
        pass 

    @abstractmethod
    def exec(self, input_image: ndarray) -> ndarray:
        pass



class Predictor(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self):
        pass



class MaskProcessing(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, mask_image: ndarray):
        pass


class GraphProcessing(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, input_image: ndarray, distance_image: ndarray, paths: dict, intersection_points: dict):
        pass


class PathsProcessing(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, mask_image: ndarray, paths_ends: list):
        pass


class PathsExcludedProcessing(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, paths, graph, intersections_points):
        pass



class Smoothing(ABC):
    
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, paths, key):
        pass



class CrossingLayout(ABC):
    
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, splines, paths, nodes, candidate_nodes, input_image, colored_mask):
        pass



class OutputMask(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, splines, shape) -> ndarray:
        pass


