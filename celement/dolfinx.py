"""Functions for coupling using DOLFINx."""

import dolfinx
import numpy as np


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
    space = function.function_space

    trace_function = dolfinx.fem.Function(trace_space)

    cmap = trace_space.mesh.topology.index_map(trace_space.mesh.topology.dim)
    num_cells = cmap.size_local + cmap.num_ghosts
    interpolation_data = dolfinx.fem.create_interpolation_data(
        trace_space, space, np.arange(num_cells, dtype=np.int32), 1e-2
    )
    trace_function.interpolate_nonmatching(
        function, np.arange(num_cells, dtype=np.int32), interpolation_data
    )
    return trace_function
