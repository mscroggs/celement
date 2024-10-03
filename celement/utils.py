"""Utility functions."""

import typing

import dolfinx
import numpy as np
import numpy.typing as npt


def get_containing_cells(
    points: npt.NDArray[np.floating], domain: dolfinx.mesh.Mesh
) -> typing.List[int]:
    """Get the cells that each point is in.

    Args:
        point: The points
        mesh: The mesh

    Returns:
        List of cell indices
    """
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)

    cells = []
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)
    for i, point in enumerate(points):
        assert len(colliding_cells.links(i)) > 0
        cells.append(colliding_cells.links(i)[0])
    return cells
