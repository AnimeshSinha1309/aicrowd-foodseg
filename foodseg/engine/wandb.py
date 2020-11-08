"""
Weights and Biases integration hooks for storing the metrics in
the Detectron model.
"""
# See https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py
# for customizing the model.

import wandb

from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.engine import DefaultTrainer
from detectron2.utils.events import CommonMetricPrinter, TensorboardXWriter


class WandbWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, window_size: int = 20, **kwargs):
        """
        Initializes the event writer

        :param window_size (int): the scalars will be median-smoothed by this window size
        :param kwargs: other arguments passed to `wandb.init(...)`
        """
        self._window_size = window_size
        if 'sync_tensorboard' not in kwargs:
            kwargs['sync_tensorboard'] = True
        wandb.init(**kwargs)
        self._log_dict = {}

    def write(self):
        """
        Logs the scalars to WandB
        """
        if wandb.run is None:
            return

        storage = get_event_storage()
        self._log_dict = {}
        for key, value in storage.latest_with_smoothing_hint(self._window_size).items():
            self._log_dict[key] = value
        wandb.log(self._log_dict, step=storage.iter)

    @property
    def previous_log(self):
        """
        Returns the logs of the previous state

        :returns: dict, log object from most recent iteration
        """
        return self._log_dict


class WandbTrainer(DefaultTrainer):
    """
    Adds hooks and visualization logic to the default trainer
    """

    def __init__(self, cfg, project="aicrowd-foodseg"):
        """
        Creates the WandbTrained object
        :param config: CfgNode, model configuration file
        """
        self.project = project
        super().__init__(cfg)

    def build_writers(self):
        """
        Registers the list of log writers
        """
        return [
            CommonMetricPrinter(self.max_iter),
            WandbWriter(config=self.cfg, job_type='train', project=self.project),
            TensorboardXWriter(log_dir='logs'),
        ]
