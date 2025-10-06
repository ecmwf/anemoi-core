# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import yaml

from anemoi.models.data_structure import build_data_handler
from anemoi.models.data_structure import build_sample_provider

logging.basicConfig(level="DEBUG")

if __name__ == "__main__":
    config_data_handler = yaml.safe_load(
        """data_handler:
            training:
                start: 1979-01-01
                end: 2021
            validation:
                start: 2022
                end: 2023-12-31
            # search_path: null
            sources:

              - dataset: aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8
                data_group: era5_o96
                extra:
                  normalisation:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  graph:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  other_config:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  normalisation: "normalisation-config-todo"

              - dataset: aifs-ea-an-oper-0001-mars-o48-1979-2022-6h-v6
                data_group: era5_o48
                extra:
                  normalisation:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  graph:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  other_config:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  normalisation: "normalisation-config-todo"

              # unused in this example: era5 n320 resolution
              - dataset: aifs-ea-an-oper-0001-mars-n320-1979-2023-6h-v8
                data_group: era5_highres
                extra:
                  normalisation:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  graph:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  other_config:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  normalisation: "normalisation-config-todo"

              # unused in this example
              - dataset: cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1
                data_group: cerra
                extra:
                  normalisation:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  graph:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  other_config: {}

              # unused in this example
              - dataset: observations-...
                data_group: iasi

              # unused in this example
              - dataset: observations-...
                data_group: metar

           """
    )
    config_sample_provider = yaml.safe_load(
        """sample_provider:
              land:
                  data_group: "era5_o48"
                  variables:
                    forcings: ["2t"]
                    prognostics: ["10u", "10v"]
                    diagnostics: ["tp"]
                  dimensions: ["offsets", "ensembles", "values", "variables"]
                  offsets: ["-6h", "+0h", "+6h"]
              atmo:
                  data_group: "era5_o96"
                  variables:
                    forcings: ["z_500", "z_850"]
                    prognostics: ["q_500", "q_850", "q_925"]
                    diagnostics: ["u_100", "v_100"]
                  dimensions: [["offsets"], "ensembles", "values", "variables"]
                  offsets: ["-6h", "+0h", "+6h"]
                  extra:
                    normalisation: # override the one in data_handler
                      {default: 'mean-std', 'mean-std': [u_100, v_100]}
                    more_config: {}
           """
    )

    data_handler = build_data_handler(config_data_handler["data_handler"], "training")
    print("Data_handler : ", data_handler)
    sp = build_sample_provider(config_sample_provider["sample_provider"], data_handler=data_handler)

    print(sp)
    print("************************")
    for key, value in sp.static.items():
        print(f"-> Static for {key}: {value}")
    print("************************")
    i = 0
    for key, value in sp[i].items():
        print(f"-> Sample data at [{i}] for {key}: {value}")
