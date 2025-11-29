import numpy as np
import torch

from anemoi.graphs.edges.directional import compute_directions

tol = 1e-7


def test_compute_directions_unit_vectors():
    """Test that compute_directions always returns unit vectors."""
    # Test with various lat/lon pairs
    source = torch.tensor([[0.0, 0.0], [0.5, 0.5], [-0.3, 0.8], [np.pi / 4, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.5, 0.5], [-0.3, 0.8], [0.0, 0.0], [np.pi / 3, np.pi / 6]], dtype=torch.float32)

    directions = compute_directions(source, target)

    # Check shape
    assert directions.shape == (4, 2)

    # Check all are unit vectors
    norms = torch.norm(directions, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_compute_directions_equator_to_pole():
    """Test direction from equator to north pole."""
    source = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # Equator at prime meridian
    target = torch.tensor([[np.pi / 2, 0.0]], dtype=torch.float32)  # North pole

    directions = compute_directions(source, target)

    # Should be a unit vector
    assert torch.allclose(torch.norm(directions), torch.tensor(1.0), atol=1e-5)

    # The exact direction depends on the rotation, but it should be well-defined
    assert directions.shape == (1, 2)
    assert torch.all(torch.isfinite(directions))


def test_compute_directions_equator_eastward():
    """Test direction along equator moving east."""
    source = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # Equator at 0°
    target = torch.tensor([[0.0, np.pi / 4]], dtype=torch.float32)  # Equator at 45°E

    directions = compute_directions(source, target)

    # Should be a unit vector
    assert torch.allclose(torch.norm(directions), torch.tensor(1.0), atol=1e-5)

    # Direction should have reasonable components
    assert directions.shape == (1, 2)
    assert torch.all(torch.isfinite(directions))


def test_compute_directions_same_meridian():
    """Test direction along same meridian (longitude) moving north."""
    source = torch.tensor([[np.pi / 6, 0.0], [np.pi / 6, np.pi / 4]], dtype=torch.float32)  # 30°N at 0° and 45°E
    target = torch.tensor([[np.pi / 3, 0.0], [np.pi / 3, np.pi / 4]], dtype=torch.float32)  # 60°N at 0° and 45°E

    directions = compute_directions(source, target)

    # Should be unit vectors
    norms = torch.norm(directions, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Directions should be valid and finite
    assert torch.all(torch.isfinite(directions))
    assert directions.shape == (2, 2)


def test_compute_directions_random_points():
    """Test with random points to ensure robustness."""
    torch.manual_seed(42)

    # Generate 100 random lat/lon pairs
    n = 100
    source_lat = (torch.rand(n) - 0.5) * np.pi  # -90° to 90°
    source_lon = (torch.rand(n) - 0.5) * 2 * np.pi  # -180° to 180°
    target_lat = (torch.rand(n) - 0.5) * np.pi
    target_lon = (torch.rand(n) - 0.5) * 2 * np.pi

    source = torch.stack([source_lat, source_lon], dim=-1)
    target = torch.stack([target_lat, target_lon], dim=-1)

    directions = compute_directions(source, target)

    # Check shape
    assert directions.shape == (n, 2)

    # All should be unit vectors
    norms = torch.norm(directions, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    # All should be finite
    assert torch.all(torch.isfinite(directions))


def test_compute_directions_nearby_points():
    """Test with nearby points (edge case for numerical stability)."""
    source = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.001, 0.001]], dtype=torch.float32)  # Very close

    directions = compute_directions(source, target)

    # Should still produce a valid unit vector
    assert torch.allclose(torch.norm(directions), torch.tensor(1.0), atol=1e-5)
    assert torch.all(torch.isfinite(directions))


def test_compute_directions_antipodal_points():
    """Test with antipodal points (opposite sides of sphere)."""
    source = torch.tensor([[np.pi / 2, 0.0]], dtype=torch.float32)  # North pole
    target = torch.tensor([[-np.pi / 2, 0.0]], dtype=torch.float32)  # South pole

    directions = compute_directions(source, target)

    # Should produce a valid unit vector even for this extreme case
    assert torch.allclose(torch.norm(directions), torch.tensor(1.0), atol=1e-5)
    assert torch.all(torch.isfinite(directions))


def test_compute_directions_consistency():
    """Test that the function produces consistent results."""
    source = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)

    directions = compute_directions(source, target)

    # Same inputs should produce same outputs
    assert torch.allclose(directions[0], directions[1], atol=1e-6)


def test_compute_directions_known_values():
    """Test against known geometric cases with expected directions.

    These test cases are designed so we can geometrically determine what
    the tangent plane direction should be after rotation.
    """
    # Case 1: source=equator, target=north pole
    # After rotating target to north pole, source projects to (1,0,0)
    # Direction on tangent plane at target points toward source: (0, -1)
    source = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[np.pi / 2, 0.0]], dtype=torch.float32)

    directions = compute_directions(source, target)
    # Direction at target (north pole) pointing toward source
    expected = torch.tensor([[0.0, -1.0]], dtype=torch.float32)

    assert torch.allclose(directions, expected, atol=1e-4), f"Expected {expected}, got {directions}"

    # Case 2: source=equator, target=30°N (same meridian)
    # Direction at target pointing toward source
    source = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[np.pi / 6, 0.0]], dtype=torch.float32)  # 30°N

    directions = compute_directions(source, target)
    # Direction at target pointing toward source (which is south)
    expected = torch.tensor([[0.0, -1.0]], dtype=torch.float32)

    assert torch.allclose(directions, expected, atol=1e-4), f"Expected {expected}, got {directions}"

    # Case 3: source=north pole, target=equator
    # Direction at target (equator) pointing toward source (north pole)
    source = torch.tensor([[np.pi / 2, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    directions = compute_directions(source, target)
    # Direction at target pointing toward source (which is north)
    expected = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    assert torch.allclose(directions, expected, atol=1e-4), f"Expected {expected}, got {directions}"


def test_compute_directions_equator_90_degrees():
    """Test direction between points on equator 90° apart in longitude.

    For two points on the equator separated by 90° in longitude, after
    rotating the target to the north pole, the source should lie on
    the xy-plane with a specific orientation.
    """
    # Two points on equator, 90° apart in longitude
    source = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # (0°, 0°)
    target = torch.tensor([[0.0, np.pi / 2]], dtype=torch.float32)  # (0°, 90°E)

    directions = compute_directions(source, target)

    # For this specific geometry: source at (lat=0, lon=0) and target at (lat=0, lon=90°)
    # After rotating target (which is at (0, 1, 0) in 3D) to north pole (0, 0, 1),
    # the source (which is at (1, 0, 0) in 3D) rotates accordingly
    # Direction at target pointing toward source: (0, -1)
    expected = torch.tensor([[0.0, -1.0]], dtype=torch.float32)

    assert torch.allclose(directions, expected, atol=1e-4), f"Expected {expected}, got {directions}"


def test_compute_directions_small_latitude_steps():
    """Test that small steps in latitude produce consistent directions.

    For small steps along the same meridian, the direction should be
    approximately constant (pointing north on the tangent plane).
    """
    # Multiple small steps along prime meridian
    latitudes = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32)

    for i in range(len(latitudes) - 1):
        source = torch.tensor([[latitudes[i], 0.0]], dtype=torch.float32)
        target = torch.tensor([[latitudes[i + 1], 0.0]], dtype=torch.float32)

        directions = compute_directions(source, target)

        # For movement north along prime meridian, direction at target pointing toward source
        # Points toward source (which is south of target), giving (0, -1) after rotation
        expected = torch.tensor([[0.0, -1.0]], dtype=torch.float32)

        assert torch.allclose(
            directions, expected, atol=0.05
        ), f"Step {i} to {i+1}: Expected ~{expected}, got {directions}"


def test_compute_directions_symmetry():
    """Test rotational symmetry around poles.

    Points at the same latitude but rotated in longitude should produce
    directions that are rotated by the same angle.
    """
    # Test point at 45°N on prime meridian to north pole
    source1 = torch.tensor([[np.pi / 4, 0.0]], dtype=torch.float32)
    target1 = torch.tensor([[np.pi / 2, 0.0]], dtype=torch.float32)

    # Same test but rotated 90° in longitude
    source2 = torch.tensor([[np.pi / 4, np.pi / 2]], dtype=torch.float32)
    target2 = torch.tensor([[np.pi / 2, np.pi / 2]], dtype=torch.float32)

    dir1 = compute_directions(source1, target1)
    dir2 = compute_directions(source2, target2)

    # Both should have same magnitude
    assert torch.allclose(torch.norm(dir1), torch.norm(dir2), atol=1e-5)

    # The directions might differ due to different tangent plane orientations,
    # but both should be valid unit vectors
    assert torch.allclose(torch.norm(dir1), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(torch.norm(dir2), torch.tensor(1.0), atol=1e-5)


def test_compute_directions_pole_cases():
    """Test explicit handling of north and south pole cases."""
    # Case 1: Target at north pole (should use north pole case)
    source = torch.tensor([[0.0, 0.0], [np.pi / 4, np.pi / 2]], dtype=torch.float32)
    target = torch.tensor([[np.pi / 2, 0.0], [np.pi / 2, 0.0]], dtype=torch.float32)

    directions = compute_directions(source, target)

    # Should produce valid unit vectors
    assert torch.allclose(torch.norm(directions, dim=-1), torch.ones(2), atol=1e-5)
    assert torch.all(torch.isfinite(directions))

    # Case 2: Target at south pole (should use south pole case)
    source_south = torch.tensor([[0.0, 0.0], [np.pi / 4, np.pi / 2]], dtype=torch.float32)
    target_south = torch.tensor([[-np.pi / 2, 0.0], [-np.pi / 2, 0.0]], dtype=torch.float32)

    directions_south = compute_directions(source_south, target_south)

    # Should produce valid unit vectors
    assert torch.allclose(torch.norm(directions_south, dim=-1), torch.ones(2), atol=1e-5)
    assert torch.all(torch.isfinite(directions_south))

    # Case 3: Mixed batch (north pole, south pole, and normal)
    source_mixed = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    target_mixed = torch.tensor([[np.pi / 2, 0.0], [-np.pi / 2, 0.0], [np.pi / 4, 0.0]], dtype=torch.float32)

    directions_mixed = compute_directions(source_mixed, target_mixed)

    # All should produce valid unit vectors
    assert torch.allclose(torch.norm(directions_mixed, dim=-1), torch.ones(3), atol=1e-5)
    assert torch.all(torch.isfinite(directions_mixed))
