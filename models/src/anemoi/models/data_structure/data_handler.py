import datetime

import numpy as np

from anemoi.datasets import open_dataset
from anemoi.models.data_structure.structure import deep_freeze_dict
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
        self._dataschema = dict(
            type="box",
            content=dict(
                latitudes=dict(type="tensor", content=None),
                longitudes=dict(type="tensor", content=None),
                timedeltas=dict(type="tensor", content=None),
                data=dict(type="tensor", content=None),
                _anemoi_schema=True,
            ),
        )

        variables = [f"{group}.{v}" for v in variables]
        self.variables = variables

        self.ds = open_dataset(dataset=self.config["dataset"], select=self.variables)
        # TODO: read from ds
        self._dimensions_order = tuple(["variables", "ensembles", "values"])

        self.frequency = frequency_to_timedelta(self.ds.frequency)
        self.statistics = deep_freeze_dict(self.ds.statistics[group])
        self.name_to_index = self.ds.name_to_index[group]
        self.metadata = self.ds.metadata.get(group, {})

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

    def reference_date(self, item=None):
        assert item is not None, item
        date = self.ds.dates[item]
        if isinstance(date, np.datetime64):
            date = date.tolist()
        assert isinstance(date, datetime.datetime), f"Expected datetime, got {type(date)}: {date}"
        return date

    def reference_date_str(self, item=None):
        return self.reference_date(item).strftime("%Y-%m-%d %H:%M:%S")

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

    def latitudes_longitudes(self, item=None):
        lats = self.latitudes(item)
        longs = self.longitudes(item)
        return np.array([lats, longs]).T

    def _get_static(self, request):
        if request is None:
            request = {}
        request["content"] = [
            "name_to_index",
            "statistics",
            "normaliser",
            "extra",
            "metadata",
            "number_of_features",
            "dimensions_order",
        ]

        return self._get(request, None)

    def _get_item(self, request, item):
        if request is None:
            request = {}
        request["content"] = request.get(
            "content",
            [
                "data",
                "latitudes",
                "longitudes",
                "timedeltas",
                "_reference_date_str",
                # "dimensions_order",
            ],
        )
        return self._get(request, item)

    def _get(self, request, item: int):

        assert isinstance(item, (type(None), int, np.integer)), f"Expected integer for item, got {type(item)}: {item}"
        assert isinstance(request, dict), f"Expected dict for request, got {type(request)}: {request}"

        content = request["content"]
        assert isinstance(content, (list, tuple)), f"Expected list for content, got {type(content)}: {content}"
        sharder = request.get("sharder", None)
        del request

        ACTIONS = {
            "data": self.__getitem__,
            "latitudes": self.latitudes,
            "longitudes": self.longitudes,
            "latitudes_longitudes": self.latitudes_longitudes,
            "timedeltas": self.timedeltas,
            "_reference_date": self.reference_date,
            "_reference_date_str": self.reference_date_str,
            "shape": lambda x: self._shape,
            "name_to_index": lambda x: self.name_to_index,
            "statistics": lambda x: self.statistics,
            "normaliser": lambda x: self._normaliser_config,
            "imputer": lambda x: self._imputer_config,
            "extra": lambda x: self._extra_config,
            "metadata": lambda x: self.metadata,
            "number_of_features": lambda x: len(self.name_to_index),
            "dimensions_order": lambda x: self._dimensions_order,
        }

        box = {r: ACTIONS[r](item) for r in content}

        class DummySharder:
            def shard_latitudes(self, latitudes):
                return latitudes

            def shard_longitudes(self, longitudes):
                return longitudes

            def shard_latitudes_longitudes(self, latitudes_longitudes):
                return latitudes_longitudes

            def shard_data(self, data, dimensions_order: list[str] | tuple[str] = None):
                return data

        if sharder is None:
            sharder = DummySharder()

        if "latitudes" in box:
            box["latitudes"] = sharder.shard_latitudes(box["latitudes"])
        if "longitudes" in box:
            box["longitudes"] = sharder.shard_longitudes(box["longitudes"])
        if "latitudes_longitudes" in box:
            box["latitudes_longitudes"] = sharder.shard_latitudes_longitudes(box["latitudes_longitudes"])
        if "data" in box:
            box["data"] = sharder.shard_data(box["data"], dimensions_order=self._dimensions_order)

        return box

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

    @property
    def dataset_name(self):
        return self.config["dataset"]

    def __repr__(self):
        return f"DataHandler {self.dataset_name} @ {self.group} [{', '.join(self.variables)}]"
