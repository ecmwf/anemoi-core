# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import yaml

from anemoi.models.data_structure import build_data_handler
from anemoi.models.data_structure import build_sample_provider

# logging.basicConfig(level="DEBUG")


def test_gridded():
    config_data_handler = yaml.safe_load(
        """data_handler:
            training:
                start: 1979-01-01
                end: 2021
            validation:
                start: 2022
                end: 2023-12-31
            search_path: null
            sources:

              - dataset: aifs-ea-an-oper-0001-mars-20p0-2016-2016-6h-v1
                data_group: era5_20
                extra_configs:
                  normalisation:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  graph:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  other_config:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  normalisation: "normalisation-config-todo"

              - dataset: aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8
                data_group: era5_o96
                extra_configs:
                  normalisation:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  graph:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  other_config:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  normalisation: "normalisation-config-todo"

              - dataset: aifs-ea-an-oper-0001-mars-o48-1979-2022-6h-v6
                data_group: era5_o48
                extra_configs:
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
                extra_configs:
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
                extra_configs:
                  normalisation:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  graph:
                    {default: 'mean-std', 'mean-std': [tp, 2t]}
                  other_config: {}
           """
    )
    config_sample_provider = yaml.safe_load(
        """sample_provider:
              dictionary:
                # land:
                #     data_group: "era5_o48"
                #     variables:
                #       forcings: ["2t"]
                #       prognostics: ["10u", "10v"]
                #       diagnostics: ["tp"]
                #     dimensions: ["offsets", "ensembles", "values", "variables"]
                #     offsets: ["-6h", "+0h", "+6h"]
                # atmo:
                #     data_group: "era5_o96"
                #     variables:
                #       forcings: ["z_500", "z_850"]
                #       prognostics: ["q_500", "q_850", "q_925"]
                #       diagnostics: ["u_100", "v_100"]
                #     dimensions: [["offsets"], "ensembles", "values", "variables"]
                #     offsets: ["-6h", "+0h", "+6h"]
                #     extra_configs:
                #       normalisation: # override the one in data_handler
                #         {default: 'mean-std', 'mean-std': [u_100, v_100]}
                #       more_config: {}
                fields:
                    data_group: "era5_20"
                    variables:
                      forcings: ["2t"]
                      prognostics: ["t_500", "t_850"]
                      diagnostics: ["z_500"]
                    dimensions: ["offsets", "ensembles", "values", "variables"]
                    offsets: ["-6h", "+0h"]
                fields2:
                    data_group: "era5_20"
                    variables:
                      forcings: ["2t"]
                      diagnostics: ["z_500"]
                    dimensions: ["offsets", "ensembles", "values", "variables"]
                    offsets: ["+0h", "+6h"]
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


def test_observations():
    config_data_handler = yaml.safe_load(
        """data_handler:
            training:
                start: 1979-01-01
                end: 2021
            validation:
                start: 2022
                end: 2023-12-31
            search_path: null
            sources:
              # unused in this example
              - dataset:
                   dataset: observations-testing-2018-2018-6h-v1
                   frequency: 3h
                data_group: iasi

              - dataset:
                   dataset: observations-testing-2018-2018-6h-v1
                   frequency: 3h
                data_group: amsr2_h180
           """
    )
    config_sample_provider = yaml.safe_load(
        """sample_provider:
              frequency: 3h
              dictionary:
                # fields:
                #     data_group: "era5_20"
                #     variables:
                #       forcings: ["2t"]
                #       prognostics: ["t_500", "t_850"]
                #       diagnostics: ["z_500"]
                #     dimensions: ["offsets", "ensembles", "values", "variables"]
                #     offsets: ["-6h", "+0h", "+6h"]
                metar:
                    data_group: "amsr2_h180"
                    variables:
                      forcings: ["rawbt_1", "rawbt_2"]
                      prognostics: ["rawbt_3"]
                      diagnostics: ["rawbt_4"]
                    dimensions: [["offsets"], "values", "variables"]
                    offsets:
                      input: ["-3h", "+0h"]
                      # input: ["-6h", "+0h"]
                      target: ["+3h"]
                    extra_configs:
                      normalisation: # override the one in data_handler
                        {default: 'mean-std', 'mean-std': [u_100, v_100]}
                      more_config: {}
           """
    )

    data_handler = build_data_handler(config_data_handler["data_handler"], "training")
    print("Data_handler : ", data_handler)
    sp = build_sample_provider(config_sample_provider["sample_provider"], data_handler=data_handler)

    print(len(sp))

    print(sp)
    print("************************")
    for key, value in sp.static.items():
        print(f"-> Static for {key}: {value}")
    print("************************")
    i = 0
    for key, value in sp[i].items():
        print(f"-> Sample data at [{i}] for {key}: {value}")


if __name__ == "__main__":
    test_gridded()
    test_observations()
