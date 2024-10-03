import numpy as np


def get_coordinates_ordering(coords: np.ndarray) -> np.ndarray:
    """Sort node coordinates by latitude and longitude.

    Parameters
    ----------
    coords : np.ndarray of shape (N, 2)
        The node coordinates, with the latitude in the first column and the
        longitude in the second column.

    Returns
    -------
    np.ndarray
        The order of the node coordinates to be sorted by latitude and longitude.
    """
    # Get indices to sort points by lon & lat in radians.
    index_latitude = np.argsort(coords[:, 1])
    index_longitude = np.argsort(coords[index_latitude][:, 0])[::-1]
    node_ordering = np.arange(coords.shape[0])[index_latitude][index_longitude]
    return node_ordering
