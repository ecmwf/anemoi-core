# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import time

import yaml

from anemoi.models.data_structure import build_sample_provider


def wrapped_build_sample_provider(*args, **kwargs):
    print("start building SampleProvider")
    t_start = time.time()
    sp = build_sample_provider(*args, **kwargs)
    t_elapsed = time.time() - t_start
    print(f"SampleProvider built in {t_elapsed:.3f} seconds")
    return sp


def wrapped_getitem(sample_provider, index):
    t_start = time.time()
    sample = sample_provider[index]
    t_elapsed = time.time() - t_start
    print(f"Sample at index {index} retrieved in {t_elapsed:.3f} seconds")
    return sample


def test_gridded():
    config_sample_provider = yaml.safe_load(
        """sample_provider:
              groups:
                # era5_o48:
                #     variables:
                #       forcings: ["2t"]
                #       prognostics: ["10u", "10v"]
                #       diagnostics: ["tp"]
                #     dimensions: ["offsets", "ensembles", "values", "variables"]
                #     offsets: ["-6h", "+0h", "+6h"]
                # era5_o96:
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
                era5_20:
                    variables:
                      forcings: ["2t"]
                      prognostics: ["t_500", "t_850"]
                      diagnostics: ["z_500"]
                    dimensions: ["offsets", "ensembles", "values", "variables"]
                    offsets: ["-6h", "+0h"]

                era5_20_bis:
                    variables:
                      forcings: ["2t"]
                      diagnostics: ["z_500"]
                    dimensions: ["offsets", "ensembles", "values", "variables"]
                    offsets: ["+0h", "+6h"]
              data_handler:
                training:
                    start: 2016-12-19
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

                  - dataset: aifs-ea-an-oper-0001-mars-20p0-2016-2016-6h-v1
                    data_group: era5_20_bis
                    extra_configs:
                      normalisation:
                        {default: 'mean-std', 'mean-std': [tp, 2t]}
                      graph:
                        {default: 'mean-std', 'mean-std': [tp, 2t]}
                      other_config:
                        {default: 'mean-std', 'mean-std': [tp, 2t]}
                      normalisation: "normalisation-config-todo"

               """
    )

    sp = wrapped_build_sample_provider(config_sample_provider["sample_provider"], kind="training")
    print("GRIDDED ************************")
    print(sp)
    for key, value in sp.static.items():
        print(f"GRIDDED -> Static for {key}: {value}")
    print("GRIDDED ************************")

    for i in [0, 5]:
        sample = wrapped_getitem(sp, i)

        for key, value in sample.items():
            print(f"GRIDDED -> Sample data at [{i}] for {key}: {value}")


def test_observations():
    cfg = yaml.safe_load(
        """sample_provider:
             frequency: 3h
             groups:
                msg_combined_seviri_o256:
                  variables:
                    forcings: ["cos_latitude", "sin_latitude"]
                    prognostics: []
                    diagnostics: []
                  dimensions: [["offsets"], "values", "variables"]
                  offsets: ["-6h", "+0h"]
                    # input: ["-3h", "+0h"]
                    # target: ["+3h"]
                  extra_configs:
                    normalisation: # override the one in data_handler
                      {default: 'mean-std', 'mean-std': [u_100, v_100]}
                    more_config: {}
             data_handler:
               training:
                   start: 2018-11-01T12:00:00
                   end: 2021
               validation:
                   start: 2022
                   end: 2023-12-31
               search_path: null
               sources:
                 - dataset:
                      # contains:  ...
                      dataset: observations-2021-10days-6h-v1-eta
                      frequency: 3h
             aliases: "is ignored"
           """
    )["sample_provider"]

    sp = wrapped_build_sample_provider(cfg, kind="training")
    print("OBS datahandler", sp.context.data_handler)

    print(len(sp))

    print(sp)
    print("OBS ************************")
    for key, value in sp.static.items():
        print(f"OBS -> Static for {key}: {value}")
    print("OBS ************************")
    i = 0
    for key, value in wrapped_getitem(sp, i).items():
        print(f"DOP -> Sample data at [{i}] for {key}: {value}")
    wrapped_getitem(sp, 5)


def test_dop():
    with open("dop_sample_provider.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sp = wrapped_build_sample_provider(cfg, kind="training")

    print(len(sp))

    print(sp)
    print("DOP ************************")
    for key, value in sp.static.items():
        print(f"DOP -> Static for {key}: {value}")
    print("DOP ************************")
    i = 0
    for key, value in wrapped_getitem(sp, i).items():
        print(f"DOP -> Sample data at [{i}] for {key}: {value}")
    wrapped_getitem(sp, 5)

    print("Now rebuilding sample providers for training and validation")
    training = build_sample_provider(cfg, kind="training")
    validation = build_sample_provider(cfg, kind="validation")
    print(f"Number of samples for training: {len(training)}")
    print(f"Number of samples for validation: {len(validation)}")


if __name__ == "__main__":
    import sys

    if "g" in sys.argv or "--gridded" in sys.argv:
        test_gridded()
    if "o" in sys.argv or "--observations" in sys.argv:
        test_observations()
    if "d" in sys.argv or "--dop" in sys.argv:
        test_dop()
    if len(sys.argv) == 1:
        test_gridded()
        test_observations()
        test_dop()
