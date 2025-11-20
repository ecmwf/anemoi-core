from typing import Annotated
from typing import Any
from typing import Literal

from pydantic import Field
from pydantic import model_validator

from anemoi.utils.schemas import BaseModel


class SkipConnectionSchema(BaseModel):
    """Schema for skip connection residuals."""

    target_: Literal["anemoi.models.layers.residual.SkipConnection"] = Field(..., alias="_target_")
    step: int = Field(
        -1,
        description="Timestep index to use for the skip connection. "
        "Defaults to -1, which selects the most recent timestep.",
    )


class TruncatedConnectionSchema(BaseModel):
    """Schema for truncated connection residuals."""

    target_: Literal["anemoi.models.layers.residual.TruncatedConnection"] = Field(..., alias="_target_")
    data_nodes: str | None = Field(
        None,
        description="Name of the node set in the graph representing the original (full) resolution data. Required if not using file paths.",
    )
    truncation_nodes: str | None = Field(
        None,
        description="Name of the node set in the graph representing the truncated (reduced) resolution data. Required if not using file paths.",
    )
    edge_weight_attribute: str | None = Field(
        None,
        description="Optional name of the edge attribute to use as weights when projecting between data and truncation nodes. Only used if not using file paths.",
    )
    src_node_weight_attribute: str | None = Field(
        None,
        description="Optional name of an attribute on source nodes to use as multiplicative weights during projection. Only used if not using file paths.",
    )
    truncation_up_file_path: str | None = Field(
        None,
        description="Optional file path (.npz) to load the up-projection matrix from. Required if not using graph-based projection.",
    )
    truncation_down_file_path: str | None = Field(
        None,
        description="Optional file path (.npz) to load the down-projection matrix from. Required if not using graph-based projection.",
    )
    autocast: bool = Field(
        False, description="Whether to enable mixed precision autocasting during projection operations."
    )

    @model_validator(mode="after")
    def check_instantiation_method(self) -> Any:
        # Check that only one method is used: either file paths or graph-based
        file_based = self.truncation_up_file_path is not None and self.truncation_down_file_path is not None
        graph_based = self.data_nodes is not None and self.truncation_nodes is not None

        if file_based and graph_based:
            raise ValueError(
                "Specify either file paths for truncation_up_file_path and truncation_down_file_path, or data_nodes and truncation_nodes for graph-based projection, but not both."
            )

        if not file_based and not graph_based:
            raise ValueError(
                "You must specify either both file paths (truncation_up_file_path and truncation_down_file_path) or both data_nodes and truncation_nodes for graph-based projection."
            )

        if file_based:
            # If using file-based, the graph-based fields should not be set
            if (
                self.data_nodes is not None
                or self.truncation_nodes is not None
                or self.edge_weight_attribute is not None
                or self.src_node_weight_attribute is not None
            ):
                raise ValueError(
                    "When using file-based projection, do not specify data_nodes, truncation_nodes, edge_weight_attribute, or src_node_weight_attribute."
                )

        if graph_based:
            # If using graph-based, the file-based fields should not be set
            if self.truncation_up_file_path is not None or self.truncation_down_file_path is not None:
                raise ValueError(
                    "When using graph-based projection, do not specify truncation_up_file_path or truncation_down_file_path."
                )


ResidualConnectionSchema = Annotated[
    SkipConnectionSchema | TruncatedConnectionSchema,
    Field(discriminator="target_"),
]
