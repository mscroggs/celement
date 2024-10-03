"""Functions for coupling using DOLFINx."""

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


def trace(
    function: dolfinx.fem.Function, trace_space: dolfinx.fem.FunctionSpace
) -> dolfinx.fem.Function:
    """Get the trace of a function in another space.

    Args:
        function: The function
        trace_space: The space to take the trace in

    Returns:
        A function in trace_space
    """
    trace_fun = dolfinx.fem.Function(trace_space)
    trace_fun.interpolate(
        lambda x: function.eval(x.T, get_containing_cells(x.T, function.function_space.mesh))[:, 0]
    )
    return trace_fun
