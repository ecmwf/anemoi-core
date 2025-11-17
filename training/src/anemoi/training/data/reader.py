import numpy as np


class AnemoiDatasetReader:
    """Data reader for anemoi-datasets."""

    def __init__(self, dataset: str | list | dict):
        from anemoi.datasets import open_dataset

        self.data = open_dataset(dataset)

    @property
    def statistics(self) -> dict:
        return self.data.statistics

    @property
    def metadata(self) -> dict:
        return self.data.metadata()

    @property
    def supporting_arrays(self) -> dict:
        return self.data.supporting_arrays()

    @property
    def name_to_index(self) -> dict:
        return self.data.name_to_index

    @property
    def resolution(self) -> str:
        return self.data.resolution

    @property
    def missing(self) -> np.ndarray:
        return self.data.missing

    @property
    def __len__(self) -> int:
        return len(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def trajectory_ids(self) -> np.ndarray | None:
        return self.data.trajectory_ids
