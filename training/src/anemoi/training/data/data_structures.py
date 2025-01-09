# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


class SourceOfSamples:
    def __init__(self, dataset, *, inputers, normalizers, mask, data_index, forcings, freq):
        from anemoi.datasets import open_dataset
        self.dataset = open_dataset(dataset)

        self.inputers = inputers

        self.normalizers = normalizers

        self.mask = mask

        self.frequency = self.dataset.frequency # needed ?
        self.resolution = self.dataset.resolution # needed ?

        self.data_index = data_index

        self.forcings = forcings # included in data_index ?
    

class DictOfSourceOfSamples(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, dataset in self.items():
            if not isinstance(dataset, SourceOfSamples):
                self[key] = SourceOfSamples(**dataset)

    def __setitem__(self, key, value):
        if not isinstance(value, SourceOfSamples):
            value = SourceOfSamples(**value)
        return super().__setitem__(key, value)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        else:
            d = {k: self[k][key] for k in self.keys()}
            return d

class ForecasterDatasetsCollection(DictOfSourceOfSamples):
    # must have only one key
    # all data nicely rectangular
    # user manages the shift in time
    pass


class LamForecasterDatasetsCollection(DictOfSourceOfSamples):
    # must have two keys
    # lam, global
    pass

class DownscalerDatasetsCollection(DictOfSourceOfSamples):
    # must have three keys, three datasets (highres and output may be the same)
    # lowres, highres, output
    pass

class ObsDatasetsCollection(DictOfSourceOfSamples):
    # any number of keys : era, metar, noaa
    pass