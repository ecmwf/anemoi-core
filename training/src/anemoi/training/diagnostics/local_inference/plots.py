import itertools
import sys
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from anemoi.datasets import open_dataset
from anemoi.training.diagnostics.maps import Coastlines
from icecream import ic

continents = Coastlines()

rocky_mountains_box = [35, 50, -120, -100]
amazon_forest_box = [-15, 5, -75, -45]
southeast_asia_box = [-10, 20, 95, 150]
west_sahara_box = [15, 30, -20, 0]
himalayas_box = [25, 40, 75, 100]
greatbarrier_reef_box = [-25, -10, 140, 155]
eastern_us = [25, 45, -90, -70]
central_africa_box = [-10, 10, 10, 30]
european_arctic_box = [-25, 0, 75, 90]

zoomed_rocky_mountains_box = [40, 45, -115, -105]  # Central Rockies
zoomed_amazon_forest_box = [-5, 5, -75, -65]  # Central Amazon
zoomed_southeast_asia_box = [0, 10, 100, 110]  # Central Indonesia
zoomed_west_sahara_box = [20, 25, -15, -5]  # Central Western Sahara
zoomed_himalayas_box = [30, 35, 80, 90]  # Central Himalayas
zoomed_greatbarrier_reef_box = [-20, -15, 145, 150]  # Central Great Barrier Reef
zoomed_eastern_us_box = [35, 40, -85, -75]  # Appalachian region
zoomed_central_africa_box = [-5, 5, 15, 25]  # Central Congo Basin


def plot_x_y(
    ds_sample: xr.Dataset,
    list_model_variables: list,
    weather_states: list[str],
    consistent_cbar: list[str] = [
        "x",
        "y",
        "y_pred",
        "y_pred_0",
        "y_pred_1",
        "y_pred_2",
        "x_0",
        "x_1",
        "x_2",
        "y_0",
        "y_1",
        "y_2",
    ],
    title=None,
):
    overlap = list(set(consistent_cbar) & set(list_model_variables))

    minmax_weather_states = get_minmax_weather_states(
        ds_sample, weather_states, overlap
    )

    figsize = (len(list_model_variables) * 4, len(weather_states) * 3)
    fig, axs = plt.subplots(
        len(weather_states), len(list_model_variables), figsize=figsize
    )

    if len(list_model_variables) == 1:
        axs = np.array([axs]).transpose()
    if len(weather_states) == 1:
        axs = np.array([axs])

    N_axs = len(weather_states) * len(list_model_variables)
    ims = {}
    cbars = {}

    for i_ax0, weather_state in enumerate(weather_states):
        for i_ax1, model_var in enumerate(list_model_variables):

            scatter_params = dict(
                x=ds_sample[ds_sample[model_var].attrs.get("lon", None)].values,
                y=ds_sample[ds_sample[model_var].attrs.get("lat", None)].values,
                c=ds_sample[model_var].sel(weather_state=weather_state).values,
                s=75_000
                / len(ds_sample[ds_sample[model_var].attrs.get("lon", None)].values),
                alpha=1.0,
                rasterized=True,
            )

            if model_var in consistent_cbar:
                scatter_params.update(
                    vmin=minmax_weather_states[weather_state][0],
                    vmax=minmax_weather_states[weather_state][1],
                    cmap="viridis",
                )

            elif model_var == "y_diff":
                vmax = np.max(
                    np.abs(ds_sample[model_var].sel(weather_state=weather_state))
                )
                scatter_params.update(vmin=-vmax, vmax=vmax, cmap="bwr")

            else:
                vmax = np.max(ds_sample[model_var].sel(weather_state=weather_state))
                vmin = np.min(ds_sample[model_var].sel(weather_state=weather_state))
                scatter_params.update(vmin=vmin, vmax=vmax, cmap="viridis")

            ims[(i_ax0, i_ax1)] = axs[i_ax0, i_ax1].scatter(**scatter_params)

            # Colourbar
            cbars[(i_ax0, i_ax1)] = plt.colorbar(
                ims[(i_ax0, i_ax1)],
                ax=axs[i_ax0, i_ax1],
                orientation="vertical",
                pad=0.05,
            )

    for i_ax0, weather_state in enumerate(weather_states):
        axs[i_ax0, 0].set_ylabel("Latitude (°)", fontsize=12)
    for i_ax1, model_var in enumerate(list_model_variables):
        axs[-1, i_ax1].set_xlabel("Longitude (°)", fontsize=12)

    # ax esthetic
    for i_ax0, weather_state in enumerate(weather_states):
        for i_ax1, model_var in enumerate(list_model_variables):
            axs[i_ax0, i_ax1].xaxis.set_major_formatter(
                ticker.FormatStrFormatter("%d°")
            )
            axs[i_ax0, i_ax1].yaxis.set_major_formatter(
                ticker.FormatStrFormatter("%d°")
            )
            axs[i_ax0, i_ax1].tick_params(axis="both", which="major", labelsize=10)

            axs[i_ax0, i_ax1].set_title(f"{model_var} - {weather_state}")

            if "region" in ds_sample.attrs:
                axs[i_ax0, i_ax1].set_xlim(
                    ds_sample.attrs["region"][2], ds_sample.attrs["region"][3]
                )
                axs[i_ax0, i_ax1].set_ylim(
                    ds_sample.attrs["region"][0], ds_sample.attrs["region"][1]
                )
            continents.plot_continents(axs[i_ax0, i_ax1])
            axs[i_ax0, i_ax1].set_aspect("auto", adjustable=None)
            axs[i_ax0, i_ax1].grid(False)

            # Frame parameters
            axs[i_ax0, i_ax1].patch.set_edgecolor("black")
            axs[i_ax0, i_ax1].patch.set_linewidth(2)

            # Colorbar parameters
            cbars[(i_ax0, i_ax1)].outline.set_edgecolor("black")  # Black border
            cbars[(i_ax0, i_ax1)].outline.set_linewidth(1.0)
            cbars[(i_ax0, i_ax1)].ax.tick_params(labelsize=10)  # Adjust tick size

    if title:
        fig.suptitle(title, fontsize=16, y=1.0)
    else:
        fig.suptitle(
            str(ds_sample.date.dt.strftime("%Y-%m-%d").values),
            fontsize=16,
            y=1.0,
        )
    fig.tight_layout()
    return fig


def get_region_ds(ds: xr.Dataset, region_box: Union[str, list[int]] = "default"):
    """Get region dataset according to bounding box."""
    predefined_boxes = {
        "default": [40, 50, 0, 10],
        "pyrenees_alpes": [40, 50, 0, 10],
        "rocky_mountains": [35, 50, -120, -100],
        "amazon_forest": [-15, 5, -75, -45],
        "southeast_asia": [-10, 20, 95, 150],
        "west_sahara": [15, 30, -20, 0],
        "himalayas": [25, 40, 75, 100],
        "greatbarrier_reef": [-25, -10, 140, 155],
        "eastern_us": [25, 45, -90, -70],
        "central_africa": [-10, 10, 10, 30],
        "european_arctic": [-25, 0, 75, 90],
        "rocky_mountains_central": [
            40,
            45,
            -115,
            -105,
        ],
        "rocky_mountains_north": [45, 50, -115, -105],
        "rocky_mountains_south": [35, 40, -110, -100],
        "amazon_forest_central": [-5, 5, -75, -65],
        "amazon_forest_west": [-10, 0, -75, -65],
        "amazon_forest_east": [-10, 0, -55, -45],
        "southeast_asia_central": [
            0,
            10,
            100,
            110,
        ],
        "southeast_asia_mainland": [10, 20, 100, 110],
        "southeast_asia_maritime": [-5, 5, 115, 125],
        "west_sahara_central": [
            20,
            25,
            -15,
            -5,
        ],
        "west_sahara_coastal": [20, 25, -20, -10],
        "west_sahara_east": [20, 25, -10, 0],
        "himalayas_central": [30, 35, 80, 90],
        "himalayas_west": [30, 35, 75, 85],
        "himalayas_east": [25, 30, 90, 100],
        "greatbarrier_reef_central": [
            -20,
            -15,
            145,
            150,
        ],
        "greatbarrier_reef_north": [-15, -10, 145, 150],
        "greatbarrier_reef_south": [-25, -20, 150, 155],
        "eastern_us_central": [35, 40, -85, -75],
        "eastern_us_north": [40, 45, -80, -70],
        "eastern_us_south": [30, 35, -85, -75],
        "central_africa_congo": [-5, 5, 15, 25],
        "central_africa_north": [0, 10, 15, 25],
        "central_africa_south": [-10, 0, 20, 30],
    }

    if isinstance(region_box, str):
        region_box = predefined_boxes.get(region_box)
        if region_box is None:
            raise ValueError(f"Bounding box '{region_box}' is not predefined.")
    elif isinstance(region_box, list) and len(region_box) != 4:
        raise ValueError("Bounding box list must have exactly 4 elements.")

    lat_min, lat_max, lon_min, lon_max = region_box

    # Create masks for hres and lres
    mask_hres = (
        (ds["lon_hres"] >= lon_min)
        & (ds["lon_hres"] <= lon_max)
        & (ds["lat_hres"] >= lat_min)
        & (ds["lat_hres"] <= lat_max)
    )
    region_hres = ds.sel(grid_point_hres=ds.grid_point_hres.where(mask_hres, drop=True))
    if "lon_lres" in ds.variables:
        mask_lres = (
            (ds["lon_lres"] >= lon_min)
            & (ds["lon_lres"] <= lon_max)
            & (ds["lat_lres"] >= lat_min)
            & (ds["lat_lres"] <= lat_max)
        )
        region_lres = region_hres.sel(
            grid_point_lres=ds.grid_point_lres.where(mask_lres, drop=True)
        )
    else:
        region_lres = region_hres

    region_lres.attrs["region"] = region_box
    return region_lres


def get_minmax_weather_states(
    ds: xr.Dataset, weather_states: list[str], list_model_variables: list[str]
):
    """Get minmax values for each weather state across all relevant model variables."""
    minmax_weather_states = {}
    for weather_state in weather_states:
        fields_val = np.concatenate(
            [
                ds.sel(weather_state=weather_state)[model_var].values.flatten().ravel()
                for model_var in list_model_variables
                if model_var in ds.variables
            ]
        ).tolist()

        minmax_weather_states[weather_state] = [
            np.min(fields_val),
            np.max(fields_val),
        ]

    return minmax_weather_states


@dataclass
class ModelVariable:
    name: str
    fields: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    fieldname_to_index: dict
    type: str = "PhysicalField"
    caracteristics: dict = None

    def __post_init__(self):
        self.fields = (
            self.fields.detach().numpy().squeeze()
            if isinstance(self.fields, torch.Tensor)
            else self.fields.squeeze()
        )
        if len(self.fields.shape) != 2:
            sys.exit()
        return self.fields

    def get_bounded_fields(self, fields, bounding_box):
        """Get bounded fields according to bounding_box."""
        self.bounded_data = {}
        lat_min, lat_max, lon_min, lon_max = bounding_box
        latlons = np.column_stack((self.lats, self.lons))
        bounded_latlons = [
            (lat, lon)
            for (lat, lon) in latlons
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
        ]
        self.bounded_data["lats"], self.bounded_data["lons"] = zip(*bounded_latlons)
        for idx_field, field in enumerate(fields):
            self.bounded_data[field] = [
                self.fields[i, self.fieldname_to_index[field]]
                for i, (lat, lon) in enumerate(latlons)
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
            ]


def get_diff_fields(y_pred: ModelVariable, y: ModelVariable):
    """Get difference fields."""
    if not np.array_equal(y.lats, y_pred.lats) or not np.array_equal(
        y.lons, y_pred.lons
    ):
        raise ValueError("Latitude and longitude grids must be identical.")

    diff_fields = np.zeros_like(y.fields) - 100

    for varname, index in y.fieldname_to_index.items():
        if varname in y_pred.fieldname_to_index:
            pred_index = y_pred.fieldname_to_index[varname]
            # Ensure shapes are compatible for subtraction
            if y.fields[:, index].shape == y_pred.fields[:, pred_index].shape:
                diff_fields[:, index] = (
                    y.fields[:, index] - y_pred.fields[:, pred_index]
                )
            else:
                raise ValueError(f"Shape mismatch for variable {varname}")

    return ModelVariable(
        name=f"{y.name}_diff",
        fields=diff_fields,
        lats=y.lats,
        lons=y.lons,
        fieldname_to_index=y.fieldname_to_index,
        type="DifferenceField",
    )


@dataclass
class FieldsPlotter:
    """To plot weather states."""

    def plot_bounded_fields(
        self,
        list_model_variables: list[ModelVariable],
        weather_states: list[str],
        bounding_box: Union[str, list[int]] = "default",
        marker_size: int = 50,
    ):

        # Define bounding boxes if bounding_box is a string
        predefined_boxes = {
            "default": [40, 50, 0, 10],
            "pyrenees_alpes": [40, 50, 0, 10],
            "rocky_mountains": [35, 50, -120, -100],
            "amazon_forest": [-15, 5, -75, -45],
            "southeast_asia": [-10, 20, 95, 150],
            "west_sahara": [15, 30, -20, 0],
            "himalayas": [25, 40, 75, 100],
            "greatbarrier_reef": [-25, -10, 140, 155],
            "eastern_us": [25, 45, -90, -70],
            "central_africa": [-10, 10, 10, 30],
        }

        if isinstance(bounding_box, str):
            bounding_box = predefined_boxes.get(bounding_box)
            if bounding_box is None:
                raise ValueError(f"Bounding box '{bounding_box}' is not predefined.")
        elif isinstance(bounding_box, list) and len(bounding_box) != 4:
            raise ValueError("Bounding box list must have exactly 4 elements.")

        # Get bounded fields of relevant model variables
        for model_var in list_model_variables:
            model_var.get_bounded_fields(weather_states, bounding_box)

        # Get minmax values for each weather state across all relevant model variables
        self.minmax_weather_states = {}
        for weather_state in weather_states:
            fields_val = list(
                itertools.chain.from_iterable(
                    [
                        model_var.bounded_data[weather_state]
                        for model_var in list_model_variables
                        if model_var.type == "PhysicalField"
                    ]
                )
            )
            self.minmax_weather_states[weather_state] = [
                np.min(fields_val),
                np.max(fields_val),
            ]
        # Plotting
        figsize = (len(list_model_variables) * 4, len(weather_states) * 3)
        fig, axs = plt.subplots(
            len(weather_states), len(list_model_variables), figsize=figsize
        )

        if len(list_model_variables) == 1:
            axs = np.array([axs]).transpose()
        if len(weather_states) == 1:
            axs = np.array([axs])

        N_axs = len(weather_states) * len(list_model_variables)
        ims = {}
        cbars = {}

        for i_ax0, var_name in enumerate(weather_states):
            for i_ax1, model_var in enumerate(list_model_variables):

                scatter_params = dict(
                    x=model_var.bounded_data["lons"],
                    y=model_var.bounded_data["lats"],
                    c=model_var.bounded_data[var_name],
                    s=75_000 / len(model_var.bounded_data[var_name]),
                    alpha=1.0,
                    rasterized=True,
                )
                sub_title = f"{model_var.name} - {var_name}"

                if model_var.type == "PhysicalField":
                    scatter_params.update(
                        vmin=self.minmax_weather_states[var_name][0],
                        vmax=self.minmax_weather_states[var_name][1],
                        cmap="viridis",
                    )

                elif model_var.type == "NoisyField":
                    scatter_params.update(cmap="viridis")
                    sub_title = f"decile: {model_var.caracteristics['decile']}, sigma: {model_var.caracteristics['sigma'].item():.2f}, weight: {model_var.caracteristics['weight'].item():.2f}"

                elif model_var.type == "DifferenceField":
                    vmax = np.max(np.abs(model_var.bounded_data[var_name]))
                    scatter_params.update(vmin=-vmax, vmax=vmax, cmap="bwr")

                else:
                    raise ValueError("Invalid model variable type")

                ims[(i_ax0, i_ax1)] = axs[i_ax0, i_ax1].scatter(**scatter_params)
                cbars[(i_ax0, i_ax1)] = plt.colorbar(
                    ims[(i_ax0, i_ax1)], ax=axs[i_ax0, i_ax1]
                )
                axs[i_ax0, i_ax1].set_title(sub_title)
                axs[i_ax0, i_ax1].set_xlim(bounding_box[2], bounding_box[3])
                axs[i_ax0, i_ax1].set_ylim(bounding_box[0], bounding_box[1])

                continents.plot_continents(axs[i_ax0, i_ax1])
                axs[i_ax0, i_ax1].set_aspect("auto", adjustable=None)
        # fig.suptitle("Weather States", fontsize=16, y=0.97)
        fig.tight_layout()


"""
if __name__ == "__main__":

    ds_era5_o96 = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6")
    ds_od_n320 = open_dataset("aifs-od-an-oper-0001-mars-n320-2016-2023-6h-v6")

    idx = np.random.randint(0, len(ds_od_n320))
    x = ModelVariable(
        name="x",
        fields=ds_era5_o96[idx].T,
        lats=ds_era5_o96.latitudes,
        lons=ds_era5_o96.longitudes,
        fieldname_to_index={
            ds_era5_o96.variables[i]: i for i in range(len(ds_era5_o96.variables))
        },
    )
    y = ModelVariable(
        name="y",
        fields=ds_od_n320[idx].T,
        lats=ds_od_n320.latitudes,
        lons=ds_od_n320.longitudes,
        fieldname_to_index={
            ds_od_n320.variables[i]: i for i in range(len(ds_od_n320.variables))
        },
    )
    y_pred = ModelVariable(
        name="y_pred",
        fields=ds_od_n320[idx].T + 1e-6 * np.random.randn(*ds_od_n320[0].T.shape),
        lats=ds_od_n320.latitudes,
        lons=ds_od_n320.longitudes,
        fieldname_to_index={
            ds_od_n320.variables[i]: i for i in range(len(ds_od_n320.variables))
        },
    )

    y_diff = get_diff_fields(y_pred, y)

    plotter = FieldsPlotter()
    plotter.plot_bounded_fields([x, y, y_pred, y_diff], ["10u", "2t"], [-45, 0, 30, 90])

    ic("End")

    plotter.plot_bounded_fields(
        ["x", "y", "y_pred", "y_diff"], ["2t"], [-45, 45, -90, 90]
    )
    plotter.plot_bounded_fields(["x", "y"], ["10u", "2t"], [-45, 0, 30, 90])

    plt.show()
"""
