import torch

from anemoi.graphs.edges.directional import direction_vec

tol = 1e-7


def test_direction_vec():
    points = torch.tensor(
        [
            [0, 0, 1],
            [0, 0, -1],
            [0, 1, 0],
            [0, -1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [1, 1, 0],
            [-1, -1, 0],
        ],
        dtype=torch.float32,
    )
    direction = direction_vec(points, torch.tensor([[0, 0, 1]], dtype=torch.float32))
    assert direction.shape == (points.shape[0], 3)
    assert torch.all(direction[:, -1] == 0)
    assert torch.abs(torch.norm(direction, dim=1) - 1).max() < tol
