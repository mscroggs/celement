"""Plot convergence to a solution.

This example solves a problem using black box coupling and makes a plot showing the order
of convergence to the exact solution.
"""

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import matplotlib.pylab as plt
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse.linalg import LinearOperator, gmres

import celement

pi = np.pi
cos = np.cos
sin = np.sin


def u_exact(x):
    return (x[2] - 0.5) * cos(pi * x[2] / 2) * x[0] * (x[0] - 1) * x[1] * (x[1] - 1)


def grad_u_exact(x):
    return np.array(
        [
            x[0] * x[1] * (x[1] - 1) * (x[2] - 1 / 2) * cos(pi * x[2] / 2)
            + x[1] * (x[0] - 1) * (x[1] - 1) * (x[2] - 1 / 2) * cos(pi * x[2] / 2),
            x[0] * x[1] * (x[0] - 1) * (x[2] - 1 / 2) * cos(pi * x[2] / 2)
            + x[0] * (x[0] - 1) * (x[1] - 1) * (x[2] - 1 / 2) * cos(pi * x[2] / 2),
            -pi * x[0] * x[1] * (x[0] - 1) * (x[1] - 1) * (x[2] - 1 / 2) * sin(pi * x[2] / 2) / 2
            + x[0] * x[1] * (x[0] - 1) * (x[1] - 1) * cos(pi * x[2] / 2),
        ]
    )


def rhs_f(x):
    return (
        pi**2 * x[0] * x[1] * (x[0] - 1) * (x[1] - 1) * (x[2] - 1 / 2) * cos(pi * x[2] / 2) / 4
        + pi * x[0] * x[1] * (x[0] - 1) * (x[1] - 1) * sin(pi * x[2] / 2)
        - 2 * x[0] * (x[0] - 1) * (x[2] - 1 / 2) * cos(pi * x[2] / 2)
        - 2 * x[1] * (x[1] - 1) * (x[2] - 1 / 2) * cos(pi * x[2] / 2)
    )


class DolfinxSide(object):
    def __init__(self, mesh, space, lambda_space, coupling_space):
        self.mesh = mesh
        self.space = space
        self.lambda_space = lambda_space
        self.coupling_space = coupling_space

        self.u = ufl.TrialFunction(self.space)
        self.v = ufl.TestFunction(self.space)
        self.lamb = ufl.TrialFunction(self.lambda_space)
        self.mu = ufl.TestFunction(self.lambda_space)

        lambda_mesh = lambda_space.mesh
        cmap = lambda_mesh.topology.index_map(lambda_mesh.topology.dim)
        self.lambda_num_cells = cmap.size_local + cmap.num_ghosts
        self.lambda_interpolation_data = dolfinx.fem.create_interpolation_data(
            lambda_space, coupling_space, np.arange(self.lambda_num_cells, dtype=np.int32), 1e-3
        )

    def A_inverse(self, fun, apply_mass=True):
        b_form = dolfinx.fem.form(ufl.inner(fun, self.v) * ufl.dx)
        if apply_mass:
            b = dolfinx.fem.petsc.assemble_vector(b_form)
            dolfinx.fem.petsc.apply_lifting(b, [self.a_form], bcs=[[self.bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        else:
            b = dolfinx.fem.petsc.create_vector(b_form)
            b.array[:] = fun.x.array[:]

        u = dolfinx.fem.Function(self.space)
        self.solver.solve(b, u.x.petsc_vec)
        return u

    def B(self, fun):
        mu = dolfinx.fem.Function(self.lambda_space)
        mu.x.array[:] = self.b_mat @ fun.x.array
        return mu

    def B_T(self, fun):
        v = dolfinx.fem.Function(self.space)
        v.x.array[:] = self.b_mat.transpose() @ fun.x.array
        return v

    def trace(self, fun):
        return celement.dolfinx.trace(fun, self.coupling_space)

    def set_bc(self, fun, marker):
        diri_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh,
            dim=(self.mesh.topology.dim - 1),
            marker=marker,
        )
        diri_dofs = dolfinx.fem.locate_dofs_topological(
            V=self.space, entity_dim=self.mesh.topology.dim - 1, entities=diri_facets
        )
        bcf = dolfinx.fem.Function(self.space)
        bcf.interpolate(fun)
        self.bc = dolfinx.fem.dirichletbc(value=bcf, dofs=np.array(diri_dofs, dtype=np.int32))

    def set_f(self, fun):
        self.f = dolfinx.fem.Function(self.space)
        self.f.interpolate(fun)

    def set_a(self, form):
        self.a_form = dolfinx.fem.form(form)
        self.a_mat = dolfinx.fem.petsc.assemble_matrix(
            self.a_form,
            bcs=[self.bc],
        )
        self.a_mat.assemble()

    def create_solver(self, petsc_options=None):
        self.solver = PETSc.KSP().create(self.mesh.comm)

        problem_prefix = f"dolfinx_solve_{id(self)}"
        self.solver.setOptionsPrefix(problem_prefix)
        self.solver.setOperators(self.a_mat)

        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        self.solver.setFromOptions()

    def set_f_form(self, form):
        self.f_form = form

    def set_b(self, form):
        self.b_mat = dolfinx.fem.assemble_matrix(form, bcs=[self.bc]).to_scipy()

    def interpolate(self, fun):
        out = dolfinx.fem.Function(self.lambda_space)

        out.interpolate_nonmatching(
            fun, np.arange(self.lambda_num_cells, dtype=np.int32), self.lambda_interpolation_data
        )
        return out


def markers_to_meshtags(mesh, tags, markers, dim):
    entities = [dolfinx.mesh.locate_entities_boundary(mesh, dim, marker) for marker in markers]
    values = [np.full_like(entities, tag) for (tag, entities) in zip(tags, entities)]
    entities = np.hstack(entities, dtype=np.int32)
    values = np.hstack(values, dtype=np.intc)
    perm = np.argsort(entities)
    return dolfinx.mesh.meshtags(mesh, dim, entities[perm], values[perm])


class DolfinxPositiveSide(DolfinxSide):
    def __init__(
        self,
        n,
        coupling_space,
        comm=MPI.COMM_WORLD,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ):
        mesh = dolfinx.mesh.create_unit_cube(comm, n, n, n)
        space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

        fdim = mesh.topology.dim - 1
        tags = [1]
        markers = [lambda x: np.isclose(x[2], 0.0)]
        ft = markers_to_meshtags(mesh, tags, markers, fdim)

        facets = ft.find(tags[0])
        submesh, submesh_to_mesh = dolfinx.mesh.create_submesh(mesh, fdim, facets)[:2]

        l_space = dolfinx.fem.functionspace(submesh, ("Discontinuous Lagrange", 0))
        super().__init__(mesh, space, l_space, coupling_space)

        self.set_bc(
            lambda x: np.zeros_like(x[0]),
            lambda x: np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
            | np.isclose(x[2], 1.0),
        )
        self.set_f(rhs_f)
        self.set_a(ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx)
        self.set_f_form(ufl.inner(self.f, self.v) * ufl.dx)

        # Create integration measures. We take mesh to be the integration domain
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
        # Since the integration domain is mesh, we must provide a map from facets
        # in mesh to cells in submesh. This is simply the "inverse" of
        # submesh_to_mesh and can be computed as follows:
        facet_imap = mesh.topology.index_map(fdim)
        num_facets = facet_imap.size_local + facet_imap.num_ghosts
        mesh_to_submesh = np.full(num_facets, -1)
        mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
        entity_maps = {submesh: mesh_to_submesh}

        self.set_b(
            dolfinx.fem.form(-ufl.inner(self.u, self.mu) * ds(tags[0]), entity_maps=entity_maps)
        )
        self.create_solver(petsc_options)


class DolfinxNegativeSide(DolfinxSide):
    def __init__(
        self,
        n,
        coupling_space,
        comm=MPI.COMM_WORLD,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ):
        cube = dolfinx.mesh.create_unit_cube(comm, n, n, n)
        mesh = dolfinx.mesh.create_mesh(
            MPI.COMM_WORLD,
            np.array(cube.geometry.dofmap, dtype=np.int64),
            np.array([[p[0], p[1], p[2] - 1.0] for p in cube.geometry.x]),
            ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
        )
        space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        fdim = mesh.topology.dim - 1
        tags = [1]
        markers = [lambda x: np.isclose(x[2], 0.0)]
        ft = markers_to_meshtags(mesh, tags, markers, fdim)

        facets = ft.find(tags[0])
        submesh, submesh_to_mesh = dolfinx.mesh.create_submesh(mesh, fdim, facets)[:2]

        l_space = dolfinx.fem.functionspace(submesh, ("Discontinuous Lagrange", 0))
        super().__init__(mesh, space, l_space, coupling_space)

        self.set_bc(
            lambda x: np.zeros_like(x[0]),
            lambda x: np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
            | np.isclose(x[2], -1.0),
        )
        self.set_f(rhs_f)
        self.set_a(ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx)
        self.set_f_form(ufl.inner(self.f, self.v) * ufl.dx)

        # Create integration measures. We take mesh to be the integration domain
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
        # Since the integration domain is mesh, we must provide a map from facets
        # in mesh to cells in submesh. This is simply the "inverse" of
        # submesh_to_mesh and can be computed as follows:
        facet_imap = mesh.topology.index_map(fdim)
        num_facets = facet_imap.size_local + facet_imap.num_ghosts
        mesh_to_submesh = np.full(num_facets, -1)
        mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
        entity_maps = {submesh: mesh_to_submesh}

        self.set_b(
            dolfinx.fem.form(-ufl.inner(self.u, self.mu) * ds(tags[0]), entity_maps=entity_maps)
        )
        self.create_solver(petsc_options)


xs = []
ys = []
for npow in range(2, 6):
    n = 2**npow
    print(f"{n=}")
    xs.append(n)

    square2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n - 1, n - 1)
    square = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        np.array(square2d.geometry.dofmap, dtype=np.int64),
        square2d.geometry.x,
        ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))),
    )
    space = dolfinx.fem.functionspace(square, ("Discontinuous Lagrange", 0))

    positive = DolfinxPositiveSide(n, space)
    negative = DolfinxNegativeSide(n, space)

    ndofs = space.dofmap.index_map.size_global
    mu = ufl.TestFunction(space)

    pos_g = positive.trace(positive.B(positive.A_inverse(positive.f)))
    neg_g = negative.trace(negative.B(negative.A_inverse(negative.f)))
    rhs = dolfinx.fem.assemble_vector(
        dolfinx.fem.form(ufl.inner(-(pos_g - neg_g), mu) * ufl.dx)
    ).array

    def lhs(coeffs):
        lamb = dolfinx.fem.Function(space)
        lamb.x.array[:] = coeffs

        pos = positive.trace(
            positive.B(positive.A_inverse(positive.B_T(positive.interpolate(lamb)), False))
        )
        neg = negative.trace(
            negative.B(negative.A_inverse(negative.B_T(negative.interpolate(lamb)), False))
        )

        return dolfinx.fem.assemble_vector(
            dolfinx.fem.form(ufl.inner(pos + neg, mu) * ufl.dx)
        ).array

    S = LinearOperator((ndofs, ndofs), matvec=lhs)

    its = 0

    def f(x):
        global its
        its += 1

    print("Starting solve")
    sol, info = gmres(S, rhs, callback=f, maxiter=500, rtol=1e-8)
    sol_func = dolfinx.fem.Function(space)
    sol_func.x.array[:] = sol

    exact_lamb = dolfinx.fem.Function(space)
    exact_lamb.interpolate(lambda x: grad_u_exact(x)[2])
    error = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(sol_func - exact_lamb, sol_func - exact_lamb) * ufl.dx)
    )
    ys.append(error)
    print(f"h={1.0 / n}")
    print(f"error={error**0.5}")
    print(f"{its=}")
    print()

plt.plot(xs, ys, "ro-")
plt.xscale("log")
plt.yscale("log")
plt.axis("equal")
plt.xlabel("n")
plt.ylabel("$\\|\\lambda_h-\\lambda\\|$")
plt.savefig("convergence.png")
plt.clf()
