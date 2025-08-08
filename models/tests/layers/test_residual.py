#  import pytest
#  import torch
#  from torch_geometric.data import HeteroData
#
#  from anemoi.models.layers.residual import TruncationMapper
#
#
#  @pytest.fixture
#  def up_down_graph():
#      graph = HeteroData()
#      x_fine = torch.tensor([[0, 0], [0, 1]])
#      x_coarse = torch.tensor([[0, 0.3]])
#
#      edge_index_down = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)  # fine → coarse
#      edge_index_up = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)  # coarse → fine
#
#      graph["data"].x = x_fine
#      graph["hidden"].x = x_coarse
#      graph["hidden", "to", "data"].edge_index = edge_index_up
#      graph["hidden", "to", "data"].edge_attr = torch.tensor([1.0, 2.0])
#
#      graph["data", "to", "hidden"].edge_index = edge_index_down
#      graph["data", "to", "hidden"].edge_attr = torch.tensor([1.0, 2.0])
#
#      return graph
#
#
#  @pytest.fixture
#  def flat_data():
#      x = torch.randn(11, 7, 5, 2, 3)  # batch, dates, ensemble, grid, features
#      return x
#
#
#  @pytest.fixture
#  def edge_index():
#      return torch.tensor([[0, 1, 1], [1, 0, 2]])
#
#
#  def test_truncation_mapper_init(up_down_graph):
#      mapper = TruncationMapper(up_down_graph)
#      assert hasattr(mapper, "A_up")
#      assert hasattr(mapper, "A_down")
#
#
#  def test_sparse_projection(up_down_graph):
#      mapper = TruncationMapper(up_down_graph)
#      x = torch.randn(10, 2, 3)
#
#      x = mapper._sparse_projection(mapper.A_down, x)
#      assert x.shape == (10, 1, 3)
#
#
#  def test_truncate_fields(up_down_graph):
#      mapper = TruncationMapper(up_down_graph)
#      x = torch.randn(10, 2, 3)  # (batch*ensemble, grid, features)
#      x_truncated = mapper._truncate_fields(x)
#
#      assert x_truncated.shape == x.shape
#
#
#  def test_forward(up_down_graph):
#      mapper = TruncationMapper(up_down_graph)
#      x = torch.randn(5, 2, 2, 2, 3)  # (batch, dates, ensemble, grid, features)
#      x_truncated = mapper.forward(x)
#      assert x_truncated.shape == (5, 2, 2, 3)  # (batch, ensemble, coarse_grid, features)
