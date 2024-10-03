import basix.ufl
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

import celement


def test_get_containing_cells():
    cube = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    square2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    square = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        square2d.geometry.dofmap,
        square2d.geometry.x,
        ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))),
    )

    p1cube = dolfinx.fem.functionspace(cube, ("Lagrange", 1))
    p1square = dolfinx.fem.functionspace(square, ("Lagrange", 1))

    f = dolfinx.fem.Function(p1cube)
    f.interpolate(lambda x: x[1] ** 2)

    f2 = dolfinx.fem.Function(p1square)
    f2.interpolate(lambda x: x[1] ** 2)

    points = []
    values = []
    cells = []
    for i, cell in enumerate(square.geometry.dofmap):
        point = sum(square.geometry.x[p] for p in cell) / len(cell)
        points.append(point)
        cells.append(i)
        for j in cell:
            point = (square.geometry.x[j] + sum(square.geometry.x[p] for p in cell)) / (
                len(cell) + 1
            )
            points.append(point)
            cells.append(i)
    points = np.array(points)

    values = f2.eval(points, cells)
    values2 = f.eval(points, celement.get_containing_cells(points, cube))

    assert np.allclose(values, values2)
