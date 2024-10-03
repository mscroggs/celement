import basix.ufl
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

import celement

cube = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
square2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
square = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD,
    np.array(square2d.geometry.dofmap, dtype=np.int64),
    square2d.geometry.x,
    ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))),
)

p1cube = dolfinx.fem.functionspace(cube, ("Lagrange", 1))
p1square = dolfinx.fem.functionspace(square, ("Lagrange", 1))

f = dolfinx.fem.Function(p1square)
f.interpolate(lambda x: x[1])

v = ufl.TestFunction(p1square)
a = dolfinx.fem.form(ufl.inner(f, v) * ufl.dx)
vec = dolfinx.fem.assemble_vector(a)

f = dolfinx.fem.Function(p1cube)
f.interpolate(lambda x: x[1])

trace_f = celement.dolfinx.trace(f, p1square)
a = dolfinx.fem.form(ufl.inner(trace_f, v) * ufl.dx)
vec2 = dolfinx.fem.assemble_vector(a)

assert np.allclose(vec.array, vec2.array)
