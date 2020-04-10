from typing import List

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("count_n_layers")
class CountNLayers(Metric):
    """
    Compute statistics of number of layers
    """
    def __init__(self, layer_indices: List[int]) -> None:
        self._layer_counts = [0 for i in layer_indices]

    def __call__(self, n_layer: int): 
        """
        Parameters
        ----------
        n_layer: ``int``
            The number of layer selected for this sample
        """
        self._layer_counts[n_layer-1] += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated count.
        """
        if reset:
            self.reset()
        return self._layer_counts

    @overrides
    def reset(self):
        self._layer_counts = [0 for i in self._layer_counts]
