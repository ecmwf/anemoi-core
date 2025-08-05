import warnings

import numpy as np

from anemoi.datasets import open_dataset
from anemoi.utils.dates import frequency_to_timedelta


class DataHandler:
    emoji = "D"
    label = "DataHandler"

    def __init__(self, group: str, variables: list = [], sources=None):
        self.group = group

        if sources is None:
            raise ValueError("For now, sources must be provided to DataHandler. For now.")

        if self.group not in sources:
            raise ValueError(
                f"Group '{self.group}' not found in sources: available groups are {list(sources.keys())}",
            )
        self.config = sources[self.group].copy()
        self._imputer_config = self.config.get("imputer", {})
        self._normaliser_config = self.config.get("normaliser", {})
        self._extra_config = self.config.get("extra", {})

        variables = [f"{group}.{v}" for v in variables]
        self.variables = variables

        self.ds = open_dataset(dataset=self.config["dataset"], select=self.variables)

        self.frequency = frequency_to_timedelta(self.ds.frequency)
        self._statistics = self.ds.statistics[group]
        self._name_to_index = self.ds.name_to_index[group]

        if hasattr(self.ds, "shapes"):
            self._shape = self.ds.shapes[group]
        elif hasattr(self.ds, "shape"):
            if isinstance(self.ds.shape, dict):
                self._shape = self.ds.shape[group]
            elif isinstance(self.ds.shape, (list, tuple)):
                self._shape = self.ds.shape
            else:
                raise ValueError(f"Unexpected shape type {type(self.ds.shape)}: {self.ds.shape}")
        else:
            raise ValueError(f"Dataset {self.ds} does not have 'shape' or 'shapes' attribute.")

        # TODO:
        # this class is not efficient and should be refactored to make sure a chunk of data is loaded only once
        # this implies changing the VariablesSampleProvider class, compress the requests for data
        # and use pointers (views) to the data that will be loaded only once
        # but the interface using "request" may stay the same

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item: int):
        return self.ds[item][self.group]

    def latitudes(self, item=None):
        if item is None:
            return self.ds.latitudes[self.group]
        return self.ds[item].latitudes[self.group]

    def longitudes(self, item=None):
        if item is None:
            return self.ds.longitudes[self.group]
        return self.ds[item].longitudes[self.group]

    def timedeltas(self, item=None):
        if item is None:
            # not implemented ?
            timedeltas = self.ds.timedeltas[self.group]
        else:
            timedeltas = self.ds[item].timedeltas[self.group]
        return timedeltas.astype("timedelta64[s]").astype(int)

    def get_statistics(self, item=None):
        warnings.warn(
            "using statistics is here deprecated, use sample_provider.statistics instead",
            DeprecationWarning,
        )
        return self._statistics

    def get_shape(self, item=None):
        return self._shape

    def latitudes_longitudes(self, item=None):
        lats = self.latitudes(item)
        longs = self.longitudes(item)
        return np.array([lats, longs]).T

    def get_name_to_index(self, item=None):
        warnings.warn(
            "using name_to_index here is deprecated, use sample_provider.name_to_index instead",
            DeprecationWarning,
        )
        return self._name_to_index

    def _get(self, item: int, request):
        assert isinstance(item, (type(None), int, np.integer)), f"Expected integer for item, got {type(item)}: {item}"
        assert isinstance(
            request,
            (type(None), str, list, tuple),
        ), f"Expected string or list for request, got {type(request)}: {request}"

        ACTIONS = {
            None: self.__getitem__,
            "data": self.__getitem__,
            "latitudes": self.latitudes,
            "longitudes": self.longitudes,
            "latitudes_longitudes": self.latitudes_longitudes,
            "timedeltas": self.timedeltas,
            "shape": self.get_shape,
            "name_to_index": self.get_name_to_index,  # moved to sample_provider.name_to_index
            "statistics": self.get_statistics,  # moved to sample_provider.statistics
        }

        def do_action(r, item):
            action = ACTIONS.get(r)

            if action is not None:
                return action(item)

            if "." in r:
                rr, key = r.split(".")
                action = ACTIONS.get(rr)
                if action is None:
                    raise ValueError(
                        f"Unknown request '{r}' in {request}. Available requests are {list(ACTIONS.keys())}.",
                    )
                return action(item)[key]

            raise ValueError(f"Unknown request '{r}' in {request}. Available requests are {list(ACTIONS.keys())}.")

        if isinstance(request, (list, tuple)):
            dic = {}
            for r in request:
                dic[r] = do_action(r, item)
            return dic

        return do_action(request, item)

    @property
    def name_to_index(self):
        return self._name_to_index

    @property
    def statistics(self):
        return self._statistics

    @property
    def imputer(self):
        return self._imputer_config

    @property
    def normaliser(self):
        return self._normaliser_config

    @property
    def processors(self):
        processors = {}
        if len(self._normaliser_config):
            processors["normaliser"] = self._normaliser_config

        if len(self._imputer_config):
            processors["imputer"] = self._imputer_config
        return processors

    @property
    def extra(self):
        return self._extra_config

    def __repr__(self):
        return f"DataHandler {self.config['dataset']} @ {self.group} [{', '.join(self.variables)}]"
