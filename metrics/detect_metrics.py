from pathlib import Path
from utils.structures import SimpleClass
from .base_metric import BaseMetric
from .map import average_precision_per_class



class DetMetrics(SimpleClass):
    """
    This class is a utility class for computing detection metrics such as precision, recall, and mean average precision
    (mAP) of an object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (tuple of str): A tuple of strings that represents the names of the classes. Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (tuple of str): A tuple of strings that represents the names of the classes.
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection metrics.
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection metrics.
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
    """

    def __init__(self, save_dir=Path('.'), plot=False, on_plot=None, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = BaseMetric()

    def process(self, tp, conf, pred_cls, target_cls):
        """Process predicted results for object detection and update metrics."""
        results = average_precision_per_class(tp,
                               conf,
                               pred_cls,
                               target_cls,
                               plot=self.plot,
                               save_dir=self.save_dir,
                               names=self.names,
                               on_plot=self.on_plot)[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ['fitness'], self.mean_results() + [self.fitness]))