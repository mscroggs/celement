import basix.ufl
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import LinearProblem

import celement

# Problem with exact solution u = x - 2y + z
# Note: GRAD u = (1, -2, 1)


def mollify(function, space):
    m_fun = dolfinx.fem.Function(space)

    def nearest_eval(x):
        x[2] = 0
        return function.eval(
            x.T, celement.dolfinx.get_containing_cells(x.T, function.function_space.mesh)
        )[:, 0]

    m_fun.interpolate(nearest_eval)
    return m_fun


class DolfinxPositiveSide(object):
    def __init__(self, n, coupling_space):
        self.mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
        self.coupling_space = coupling_space
        self.space = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 1))

        diri_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh,
            dim=(self.mesh.topology.dim - 1),
            marker=lambda x: np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
            | np.isclose(x[2], 1.0),
        )
        diri_dofs = dolfinx.fem.locate_dofs_topological(
            V=self.space, entity_dim=2, entities=diri_facets
        )
        f = dolfinx.fem.Function(self.space)
        f.interpolate(lambda x: x[0] - 2 * x[1] + x[2])
        self.bc = dolfinx.fem.dirichletbc(value=f, dofs=np.array(diri_dofs, dtype=np.int32))

        self.u = ufl.TrialFunction(self.space)
        self.v = ufl.TestFunction(self.space)
        self.a = ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx

    def solve(self, neumann_fun):
        lamb = mollify(neumann_fun, self.space)

        L = ufl.inner(lamb, self.v) * ufl.ds
        problem = LinearProblem(
            self.a, L, bcs=[self.bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        uh = problem.solve()

        return celement.dolfinx.trace(uh, self.coupling_space)


class DolfinxNegativeSide(object):
    def __init__(self, n, coupling_space):
        cube = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
        self.mesh = dolfinx.mesh.create_mesh(
            MPI.COMM_WORLD,
            np.array(cube.geometry.dofmap, dtype=np.int64),
            np.array([[p[0], p[1], p[2] - 1.0] for p in cube.geometry.x]),
            ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
        )
        self.coupling_space = coupling_space
        self.space = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 1))

        diri_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh,
            dim=(self.mesh.topology.dim - 1),
            marker=lambda x: np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
            | np.isclose(x[2], -1.0),
        )
        diri_dofs = dolfinx.fem.locate_dofs_topological(
            V=self.space, entity_dim=2, entities=diri_facets
        )
        f = dolfinx.fem.Function(self.space)
        f.interpolate(lambda x: x[0] - 2 * x[1] + x[2])
        self.bc = dolfinx.fem.dirichletbc(value=f, dofs=np.array(diri_dofs, dtype=np.int32))

        self.u = ufl.TrialFunction(self.space)
        self.v = ufl.TestFunction(self.space)
        self.a = ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx

    def solve(self, neumann_fun):
        lamb = mollify(neumann_fun, self.space)

        L = -ufl.inner(lamb, self.v) * ufl.ds
        problem = LinearProblem(
            self.a, L, bcs=[self.bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        uh = problem.solve()

        return celement.dolfinx.trace(uh, self.coupling_space)


square2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
square = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD,
    np.array(square2d.geometry.dofmap, dtype=np.int64),
    square2d.geometry.x,
    ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))),
)
p1square = dolfinx.fem.functionspace(square, ("Lagrange", 1))

positive = DolfinxPositiveSide(5, p1square)
negative = DolfinxNegativeSide(7, p1square)

# Check that each half gives the correct solution
lamb = dolfinx.fem.Function(p1square)
lamb.interpolate(lambda x: -1.0 + x[0] * 0)
uh = positive.solve(lamb)
uh2 = negative.solve(lamb)

f = dolfinx.fem.Function(p1square)
f.interpolate(lambda x: x[0] - 2 * x[1] + x[2])

e = uh - f
error = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(e, e) * ufl.dx))
assert np.isclose(error, 0.0)
e = uh2 - f
error = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(e, e) * ufl.dx))
assert np.isclose(error, 0.0)
