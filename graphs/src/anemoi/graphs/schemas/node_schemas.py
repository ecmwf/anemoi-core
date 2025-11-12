# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging
from pathlib import Path  # noqa: TC003
from typing import Annotated
from typing import Literal
from typing import Optional

from pydantic import Field
from pydantic import PositiveFloat
from pydantic import PositiveInt

from anemoi.utils.schemas import BaseModel

LOGGER = logging.getLogger(__name__)


class AnemoiDatasetNodeSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.AnemoiDatasetNodes", "anemoi.graphs.nodes.ZarrDatasetNodes"] = Field(
        ..., alias="_target_"
    )
    "Nodes from Anemoi dataset class implementation from anemoi.graphs.nodes."
    dataset: str | list | dict  # TODO(Helen): Discuss schema with Baudouin
    "The dataset containing the nodes."


class NPZnodeSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.NPZFileNodes"] = Field(..., alias="_target_")
    "Nodes from NPZ grids class implementation from anemoi.graphs.nodes."
    npz_file: str
    "Path to the NPZ file."
    lat_key: str = Field(examples=["lat", "latitude"])
    "The key name of the latitude field."
    lon_key: str = Field(examples=["lon", "longitude"])
    "The key name of the longitude field."


class TextNodeSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.TextNodes"] = Field(..., alias="_target_")
    "Nodes from text file class implementation from anemoi.graphs.nodes."
    dataset: str | Path
    "The path to text file containing the coordinates of the nodes."
    idx_lon: int
    "The index of the longitude in the dataset."
    idx_lat: int
    "The index of the latitude in the dataset."


class XArrayNodeSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.XArrayNodes"] = Field(..., alias="_target_")
    "Nodes from xarray dataset class implementation from anemoi.graphs.nodes."
    dataset: str | Path
    "The path to xarray dataset containing the coordinates of the nodes."
    lon_key: str
    "The key name of the longitude field."
    lat_key: str
    "The key name of the latitude field."


class ReducedGaussianGridNodeSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.ReducedGaussianGridNodes"] = Field(..., alias="_target_")
    "Nodes from NPZ grids class implementation from anemoi.graphs.nodes."
    grid: Literal["o16", "o32", "o48", "o96", "o160", "o256", "o320", "n320", "o1280"]
    "Reduced gaussian grid."


class ICONNodeSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.ICONNodes"] = Field(..., alias="_target_")
    "ICON grid object from anemoi.graphs.nodes."
    name: str
    "Name of ICON grid."
    grid_filename: str
    "Name of NetCDF ICON grid file."
    max_level_multimesh: int
    "Maximum refinement level of the multi mesh."
    max_level_dataset: int
    "Maximum refinement level of the cell grid."


class ICONMeshNodeSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.nodes.ICONMultimeshNodes",
        "anemoi.graphs.nodes.ICONCellGridNodes",
    ] = Field(
        ...,
        alias="_target_",
    )
    "Mesh based on ICON grid class implementation from anemoi.graphs.nodes."
    icon_mesh: str
    "Key corresponding to the ICON mesh (cells and vertices)."


class LimitedAreaNPZFileNodesSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.LimitedAreaNPZFileNodes"] = Field(..., alias="_target_")
    "Class implementation for nodes from NPZ grids using an area of interest from anemoi.graphs.nodes."
    grid_definition_path: str
    "Path to the folder containing the grid definition files."
    resolution: str = Field(examples=["o48", "o96", "n320"])  # TODO(Helen): Discuss with Mario about the refactor
    "The grid resolution."
    reference_node_name: str  # TODO(Helen): Check that reference nodes exists in the config
    "Name of the reference nodes in the graph to consider for the Area Mask."
    mask_attr_name: Optional[str]  # TODO(Helen): Check that mask_attr_name exists in the dataset config
    "Name of a node to attribute to mask the reference nodes, if desired. Defaults to consider all reference nodes."
    margin_radius_km: PositiveFloat = Field(example=100.0)
    "Maximum distance to the reference nodes to consider a node as valid, in kilometers. Defaults to 100 km."


class IcosahedralandHealPixNodeSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.nodes.TriNodes",
        "anemoi.graphs.nodes.HexNodes",
        "anemoi.graphs.nodes.HEALPixNodes",
    ] = Field(..., alias="_target_")
    "Icohedral and HEAL Pix nodes class implementations from anemoi.graphs.nodes."
    resolution: PositiveInt
    "Refinement level of the mesh."


class LimitedAreaIcosahedralandHealPixNodeSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.nodes.LimitedAreaTriNodes",
        "anemoi.graphs.nodes.LimitedAreaHexNodes",
        "anemoi.graphs.nodes.LimitedAreaHEALPixNodes",
    ] = Field(..., alias="_target_")
    "Class implementations for Icosahedral and HEAL Pix nodes using an area of interest from anemoi.graphs.nodes."
    resolution: PositiveInt
    "Refinement level of the mesh."
    reference_node_name: str  # TODO(Helen): Discuss check that reference nodes exists in the config
    "Name of the reference nodes in the graph to consider for the Area Mask."
    mask_attr_name: Optional[str]  # TODO(Helen): Discuss check that mask_attr_name exists in the dataset config
    "Name of a node to attribute to mask the reference nodes, if desired. Defaults to consider all reference nodes."
    margin_radius_km: PositiveFloat = Field(example=100.0)
    "Maximum distance to the reference nodes to consider a node as valid, in kilometers. Defaults to 100 km."


class StretchedIcosahdralNodeSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.StretchedTriNodes"] = Field(..., alias="_target_")
    "Class implementation for nodes based on iterative refinements of an icosahedron with 2 different resolutions."
    global_resolution: PositiveInt
    "Refinement level of the mesh on the global area."
    lam_resolution: PositiveInt
    "Refinement level of the mesh on the local area."
    reference_node_name: str
    "Name of the reference nodes in the graph to consider for the Area Mask."
    mask_attr_name: Optional[str]
    "Name of a node to attribute to mask the reference nodes, if desired. Defaults to consider all reference nodes."
    margin_radius_km: PositiveFloat = Field(example=100.0)
    "Maximum distance to the reference nodes to consider a node as valid, in kilometers. Defaults to 100 km."


NodeBuilderSchemas = Annotated[
    AnemoiDatasetNodeSchema
    | NPZnodeSchema
    | TextNodeSchema
    | ICONNodeSchema
    | ICONMeshNodeSchema
    | LimitedAreaNPZFileNodesSchema
    | ReducedGaussianGridNodeSchema
    | IcosahedralandHealPixNodeSchema
    | LimitedAreaIcosahedralandHealPixNodeSchema
    | StretchedIcosahdralNodeSchema,
    Field(discriminator="target_"),
]
