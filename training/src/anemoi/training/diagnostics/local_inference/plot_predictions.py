import glob
import xarray as xr
import sys
import os
from icecream import ic
import torch
import numpy as np
import matplotlib.pyplot as plt
from anemoi.training.diagnostics.maps import Coastlines
import matplotlib.ticker as ticker
from dataclasses import dataclass
import logging
import pandas as pd

from anemoi.training.diagnostics.local_inference.plots import (
    get_region_ds,
    get_minmax_weather_states,
    plot_x_y,
)

import seaborn as sns

plt.style.use("default")
continents = Coastlines()
import itertools
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("local_inference_plotter.log"),
        logging.StreamHandler(),
    ],
)


@dataclass
class LocalInferencePlotter:
    dir_exp: str
    name_exp: str
    name_predictions_file: str

    def __post_init__(self):
        self.ds = xr.open_dataset(
            os.path.join(self.dir_exp, self.name_exp, self.name_predictions_file)
        )

        if self.ds.attrs["grid"] == "O320":
            self.grid = "O320"
            self.regions = [
                "amazon_forest",
                "european_arctic",
                "himalayas",
                "rocky_mountains",
                "west_sahara",
                "pyrenees_alpes",
                "eastern_us",
                "central_africa",
            ]
        elif self.ds.attrs["grid"] == "O1280":
            self.grid = "O1280"
            self.regions = [
                "rocky_mountains_central",
                "rocky_mountains_north",
                "rocky_mountains_south",
                "amazon_forest_central",
                "amazon_forest_west",
                "amazon_forest_east",
                "southeast_asia_central",
                "southeast_asia_mainland",
                "southeast_asia_maritime",
                "west_sahara_central",
                "west_sahara_coastal",
                "west_sahara_east",
                "himalayas_central",
                "himalayas_west",
                "himalayas_east",
                "greatbarrier_reef_central",
                "greatbarrier_reef_north",
                "greatbarrier_reef_south",
                "eastern_us_central",
                "eastern_us_north",
                "eastern_us_south",
                "central_africa_congo",
                "central_africa_north",
                "central_africa_south",
            ]
        else:
            raise ValueError(
                f"Unsupported grid type: {self.ds.attrs['grid']}. "
                "Please ensure the grid attribute is either 'O320' or 'O1280'."
            )

    def save_plot(
        self,
        list_regions,
        list_model_variables=["x_0", "y_0", "y_1", "y_pred_0", "y_pred_1"],
        weather_states=["10u", "10v", "2t", "z_500", "u_850", "v_850", "t_850"],
    ):
        with PdfPages(f"{self.dir_exp}/{self.name_exp}/all_regions_plots.pdf") as pdf:
            for region in list_regions:
                logging.info(f"Plotting region {region}")
                region_ds = get_region_ds(self.ds, region)
                region_ds.attrs["region_name"] = region

                if "sample" in region_ds.dims:
                    # Original behavior when sample dimension exists
                    for sample in range(2):
                        fig = plot_x_y(
                            ds_sample=region_ds.sel(sample=sample),
                            list_model_variables=list_model_variables,
                            weather_states=weather_states,
                            title=f"{region} - sample {sample}",
                        )
                        pdf.savefig(fig)
                        plt.close(fig)
                else:
                    # Alternative when using step and forecast_reference_time
                    sample_count = 0
                    for step in region_ds.step.values:
                        for ft in np.atleast_1d(
                            region_ds.forecast_reference_time.values
                        ):
                            if sample_count >= 2:
                                break
                            ic(region_ds.sel(step=step, forecast_reference_time=ft))
                            fig = plot_x_y(
                                ds_sample=region_ds.sel(
                                    step=step, forecast_reference_time=ft
                                ),
                                list_model_variables=list_model_variables,
                                weather_states=weather_states,
                                title=f"{region} - step {step} - forecast {pd.to_datetime(ft).strftime('%Y-%m-%d')}",
                            )
                            pdf.savefig(fig)
                            plt.close(fig)
                            sample_count += 1
                        if sample_count >= 2:
                            break

        logging.info(
            f"Plot saved successfully at {self.dir_exp}/{self.name_exp}/all_regions_plots.pdf"
        )


if __name__ == "__main__":
    dir_exp = "/home/ecm5702/scratch/aifs/checkpoint"
    name_exp = "8c8d95213c8e4df6b5784795cd6411d2"
    name_pred = "predictions.nc"

    lip = LocalInferencePlotter(dir_exp, name_exp, name_pred)
    lip.save_plot(lip.regions)
