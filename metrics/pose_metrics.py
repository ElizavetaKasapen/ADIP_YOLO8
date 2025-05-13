from pathlib import Path
import numpy as  np
from .segment_metrics import SegmentMetrics
from .base_metric import BaseMetric
from .map import average_precision_per_class

OKS_SIGMA = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

class PoseMetrics(SegmentMetrics):
    """
    Calculates and aggregates detection and pose metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (list): List of class names. Default is an empty list.

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        plot (bool): Whether to save the detection and segmentation plots.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (list): List of class names.
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        pose (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict): Dictionary to store the time taken in different phases of inference.

    Methods:
        process(tp_m, tp_boxes, conf, pred_cls, target_cls): Processes metrics over the given set of predictions.
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
    """

    def __init__(self, save_dir=Path('.'), plot=False, on_plot=None, names=()) -> None:
        super().__init__(save_dir, plot, names)
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = BaseMetric()
        self.pose = BaseMetric()

    def __getattr__(self, attr):
        """Raises an AttributeError if an invalid attribute is accessed."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def process(self, tp_boxes, tp_poses, conf, pred_cls, target_cls):
        """
        Processes the detection and pose metrics over the given set of predictions.

        Args:
            tp_boxes (list): List of True Positive boxes.
            tp_poses (list): List of True Positive keypoints.
            conf (list): List of confidence scores.
            pred_cls (list): List of predicted classes.
            target_cls (list): List of target classes.
        """

        results_pose = average_precision_per_class(tp_poses,
                                    conf,
                                    pred_cls,
                                    target_cls,
                                    plot=self.plot,
                                    on_plot=self.on_plot,
                                    save_dir=self.save_dir,
                                    names=self.names,
                                    prefix='Pose')[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)
        results_box = average_precision_per_class(tp_boxes,
                                   conf,
                                   pred_cls,
                                   target_cls,
                                   plot=self.plot,
                                   on_plot=self.on_plot,
                                   save_dir=self.save_dir,
                                   names=self.names,
                                   prefix='Box')[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        """Returns list of evaluation metric keys."""
        return [
            'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'metrics/precision(P)', 'metrics/recall(P)', 'metrics/mAP50(P)', 'metrics/mAP50-95(P)']

    def mean_results(self):
        """Return the mean results of box and pose."""
        return self.box.mean_results() + self.pose.mean_results()

    def class_result(self, i):
        """Return the class-wise detection results for a specific class i."""
        return self.box.class_result(i) + self.pose.class_result(i)

    @property
    def maps(self):
        """Returns the mean average precision (mAP) per class for both box and pose detections."""
        return self.box.maps + self.pose.maps

    @property
    def fitness(self):
        """Computes classification metrics and speed using the `targets` and `pred` inputs."""
        return self.pose.fitness() + self.box.fitness()