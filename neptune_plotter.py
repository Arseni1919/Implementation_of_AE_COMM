from GLOBALS import *


class NeptunePlotter:
    """
    Neptune + final plots
    """
    def __init__(self, plot_neptune=False, tags=None, name='check'):
        # NEPTUNE
        self.plot_neptune = plot_neptune
        self.tags = [] if tags is None else tags
        self.name = name
        self.run = {}
        self.neptune_initiated = False
        self.neptune_init()

        # FINAL PLOTS
        self.remained_coverage_dict = {}

    def neptune_init(self, params=None):
        if params is None:
            params = {}

        if self.plot_neptune:
            self.run = neptune.init(project='1919ars/ae-comm',
                                    tags=self.tags,
                                    name=f'{self.name}')

            self.run['parameters'] = params
            self.neptune_initiated = True

    def neptune_plot(self, update_dict: dict):
        if self.plot_neptune:

            if not self.neptune_initiated:
                raise RuntimeError('~[ERROR]: Initiate NEPTUNE!')

            for k, v in update_dict.items():
                self.run[k].log(v)

    def close(self):
        if self.plot_neptune and self.neptune_initiated:
            self.run.stop()

    def update_metrics(self, update_dict: dict):
        pass

    def update_metrics_and_neptune(self, update_dict: dict):
        """use it if the dict is the same for both methods"""
        self.update_metrics(update_dict)
        self.neptune_plot(update_dict)
