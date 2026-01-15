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


class TruncationGraphSchema(BaseModel):
    """Schema for graph-based truncation."""

    graph_config: dict[str, Any] | str
    down_edges_name: list[str]
    up_edges_name: list[str]
    edge_weight_attribute: str | None = None
    src_node_weight_attribute: str | None = None

    @model_validator(mode="after")
    def validate_edges(self) -> "TruncationGraphSchema":
        if len(self.down_edges_name) != 3 or len(self.up_edges_name) != 3:
            raise ValueError("down_edges_name and up_edges_name must be [src, relation, dst].")
        return self


class TruncatedConnectionSchema(BaseModel):
    """Schema for truncated connection residuals."""

    target_: Literal["anemoi.models.layers.residual.TruncatedConnection"] = Field(..., alias="_target_")
    truncation_up_file_path: str | None = Field(
        None,
        description="Optional file path (.npz) to load the up-projection matrix from. Required if not using truncation_graph.",
    )
    truncation_down_file_path: str | None = Field(
        None,
        description="Optional file path (.npz) to load the down-projection matrix from. Required if not using truncation_graph.",
    )
    truncation_matrices_path: str | None = Field(
        None,
        description="Optional base path for resolving truncation matrix file paths.",
    )
    truncation_graphs_path: str | None = Field(
        None,
        description="Optional base path for resolving truncation graph configs.",
    )
    truncation_graph: TruncationGraphSchema | None = Field(
        None,
        description="Graph-based truncation specification (graph_config + edge definitions).",
    )
    autocast: bool = Field(
        False, description="Whether to enable mixed precision autocasting during projection operations."
    )

    @model_validator(mode="after")
    def check_instantiation_method(self) -> Any:
        file_based = self.truncation_up_file_path is not None and self.truncation_down_file_path is not None
        graph_config_based = self.truncation_graph is not None

        if sum([file_based, graph_config_based]) != 1:
            raise ValueError("Specify exactly one truncation source: file paths or truncation_graph.")

        if self.truncation_matrices_path is not None and not file_based:
            raise ValueError("truncation_matrices_path requires truncation_up_file_path and truncation_down_file_path.")

        if self.truncation_graphs_path is not None and not graph_config_based:
            raise ValueError("truncation_graphs_path requires truncation_graph.")

        if file_based:
            if self.truncation_graph is not None:
                raise ValueError("When using file-based projection, do not specify truncation_graph.")
            if self.truncation_graphs_path is not None:
                raise ValueError("When using file-based projection, do not specify truncation_graphs_path.")

        if graph_config_based:
            if self.truncation_up_file_path is not None or self.truncation_down_file_path is not None:
                raise ValueError(
                    "When using truncation_graph, do not specify truncation_up_file_path or truncation_down_file_path."
                )
            if self.truncation_matrices_path is not None:
                raise ValueError("When using truncation_graph, do not specify truncation_matrices_path.")
        return self

        return self


ResidualConnectionSchema = Annotated[
    SkipConnectionSchema | TruncatedConnectionSchema,
    Field(discriminator="target_"),
]
