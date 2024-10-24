import basix.ufl
import dolfinx
import numpy as np
import pytest
import ufl
from mpi4py import MPI

import celement


@pytest.mark.parametrize(
    "cube_cell_type",
    [
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize(
    "square_cell_type",
    [
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
    ],
)
@pytest.mark.parametrize("cube_npts", [3, 5, 7])
def test_get_containing_cells(cube_cell_type, square_cell_type, cube_npts):
    cube = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, cube_npts, cube_npts, cube_npts, cube_cell_type
    )
    square2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, square_cell_type)
    square = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        square2d.geometry.dofmap,
        square2d.geometry.x,
        ufl.Mesh(basix.ufl.element("Lagrange", square_cell_type.name, 1, shape=(3,))),
    )

    p1cube = dolfinx.fem.functionspace(cube, ("Lagrange", 1))
    p1square = dolfinx.fem.functionspace(square, ("Lagrange", 1))

    f = dolfinx.fem.Function(p1cube)
    f.interpolate(lambda x: x[1])

    f2 = dolfinx.fem.Function(p1square)
    f2.interpolate(lambda x: x[1])

    pts = []
    cells = []
    for i, cell in enumerate(square.geometry.dofmap):
        point = sum(square.geometry.x[p] for p in cell) / len(cell)
        pts.append(point)
        cells.append(i)
        for j in cell:
            point = (square.geometry.x[j] + sum(square.geometry.x[p] for p in cell)) / (
                len(cell) + 1
            )
            pts.append(point)
            cells.append(i)
    points = np.array(pts)

    values = f2.eval(points, cells)
    values2 = f.eval(points, celement.dolfinx.get_containing_cells(points, cube))

    print(cells)

    for p, i, j in zip(points, values, values2):
        print(p, i, j)
    assert np.allclose(values, values2)


@pytest.mark.parametrize(
    "cube_cell_type",
    [
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize(
    "square_cell_type",
    [
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
    ],
)
@pytest.mark.parametrize("npts", [3, 5, 7])
def test_trace(cube_cell_type, square_cell_type, npts):
    cube = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, npts, npts, npts, cube_cell_type)
    square2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, npts, npts, square_cell_type)
    square = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        square2d.geometry.dofmap,
        square2d.geometry.x,
        ufl.Mesh(basix.ufl.element("Lagrange", square_cell_type.name, 1, shape=(3,))),
    )

    p1cube = dolfinx.fem.functionspace(cube, ("Lagrange", 1))
    p1square = dolfinx.fem.functionspace(square, ("Lagrange", 1))
    v = ufl.TestFunction(p1square)

    f = dolfinx.fem.Function(p1square)
    f.interpolate(lambda x: x[0] ** 4)
    coeffs = f.x.array
    a = dolfinx.fem.form(ufl.inner(f, v) * ufl.dx)
    vec = dolfinx.fem.assemble_vector(a)

    f = dolfinx.fem.Function(p1cube)
    f.interpolate(lambda x: x[0] ** 4)
    trace_f = celement.dolfinx.trace(f, p1square)
    coeffs2 = trace_f.x.array
    a = dolfinx.fem.form(ufl.inner(trace_f, v) * ufl.dx)
    vec2 = dolfinx.fem.assemble_vector(a)

    assert np.allclose(coeffs, coeffs2)
    assert np.allclose(vec.array, vec2.array)


@pytest.mark.parametrize(
    "cube_cell_type",
    [
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize(
    "square_cell_type",
    [
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
    ],
)
@pytest.mark.parametrize("cube_npts", [3, 5, 7])
def test_non_matching_trace(cube_cell_type, square_cell_type, cube_npts):
    cube = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, cube_npts, cube_npts, cube_npts, cube_cell_type
    )
    square2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, square_cell_type)
    square = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        square2d.geometry.dofmap,
        square2d.geometry.x,
        ufl.Mesh(basix.ufl.element("Lagrange", square_cell_type.name, 1, shape=(3,))),
    )

    p1cube = dolfinx.fem.functionspace(cube, ("Lagrange", 1))
    p1square = dolfinx.fem.functionspace(square, ("Lagrange", 1))
    v = ufl.TestFunction(p1square)

    f = dolfinx.fem.Function(p1square)
    f.interpolate(lambda x: x[1])
    coeffs = f.x.array
    a = dolfinx.fem.form(ufl.inner(f, v) * ufl.dx)
    vec = dolfinx.fem.assemble_vector(a)

    f = dolfinx.fem.Function(p1cube)
    f.interpolate(lambda x: x[1])
    trace_f = celement.dolfinx.trace(f, p1square)
    coeffs2 = trace_f.x.array
    a = dolfinx.fem.form(ufl.inner(trace_f, v) * ufl.dx)
    vec2 = dolfinx.fem.assemble_vector(a)

    assert np.allclose(coeffs, coeffs2)
    assert np.allclose(vec.array, vec2.array)
