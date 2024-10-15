import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import celement

# Problem with exact solution u = sin(pi*z)x(x-1)y(y-1)
# Note: GRAD u =  (
#           -2*x*y**2*z**2 + 2*x*y**2 + 2*x*y*z**2 - 2*x*y + y**2*z**2 - y**2 - y*z**2 + y,
#           -2*x**2*y*z**2 + 2*x**2*y + x**2*z**2 - x**2 + 2*x*y*z**2 - 2*x*y - x*z**2 + x,
#           -2*x**2*y**2*z + 2*x**2*y*z + 2*x*y**2*z - 2*x*y*z,
#       )
#       f = 2*x**2*y**2 - 2*x**2*y + 2*x**2*z**2 - 2*x**2 - 2*x*y**2 + 2*x*y - 2*x*z**2 + 2*x + 2*y**2*z**2 - 2*y**2 - 2*y*z**2 + 2*y

def u_exact(x):
    return np.sin(np.pi*x[2])*x[0]*(x[0]-1)*x[1]*(x[1]-1)


def grad_u_exact(x):
    return np.array([
        np.sin(np.pi * x[2]) * x[1] * (x[1] - 1) * (2 * x[0] - 1),
        np.sin(np.pi * x[2]) * x[0] * (x[0] - 1) * (2 * x[1] - 1),
        np.pi * np.cos(np.pi * x[2]) * x[0] * (x[0] - 1) * x[1] * (x[1] - 1)
    ])


def rhs_f(x):
    return -np.sin(np.pi * x[2]) * (x[1] * (x[1] - 1) * 2 + x[0] * (x[0] - 1) * 2 - np.pi**2 * x[0] * (x[0] - 1) * x[1] * (x[1] - 1))


def mollify(function, space):
    m_fun = dolfinx.fem.Function(space)

    def nearest_eval(x):
        x[2] = 0
        return function.eval(
            x.T, celement.dolfinx.get_containing_cells(x.T, function.function_space.mesh)
        )[:, 0]

    m_fun.interpolate(nearest_eval)
    return m_fun


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

    def mollify(self, fun):
        return mollify(fun, self.lambda_space)

    def A_inverse(self, fun):
        u = dolfinx.fem.Function(self.space)

        b_form = dolfinx.fem.form(ufl.inner(fun, self.v) * ufl.dx)
        b = dolfinx.fem.petsc.assemble_vector(b_form)
        dolfinx.fem.petsc.apply_lifting(b, [self.a_form], bcs=[[self.bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore

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
            self.a_form, bcs=[self.bc],
        )
        self.a_mat.assemble()

        # TODO: remove
        self.a_mat_scipy = dolfinx.fem.assemble_matrix(
            self.a_form, bcs=[self.bc],
        ).to_scipy()

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
        self.b_mat = dolfinx.fem.assemble_matrix(dolfinx.fem.form(form), bcs=[self.bc]).to_scipy()


class DolfinxPositiveSide(DolfinxSide):
    def __init__(self, n, coupling_space, comm=MPI.COMM_WORLD, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}):
        mesh = dolfinx.mesh.create_unit_cube(comm, n, n, n)
        space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        l_space = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 0))
        super().__init__(mesh, space, l_space, coupling_space)

        self.set_bc(lambda x: np.zeros_like(x[0]), lambda x: np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
            | np.isclose(x[2], 1.0),
        )
        self.set_f(rhs_f)
        self.set_a(ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx)
        self.set_f_form(ufl.inner(self.f, self.v) * ufl.dx)
        self.set_b(-ufl.inner(self.u, self.mu) * ufl.ds)
        self.create_solver(petsc_options)


class DolfinxNegativeSide(DolfinxSide):
    def __init__(self, n, coupling_space, comm=MPI.COMM_WORLD, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}):
        cube = dolfinx.mesh.create_unit_cube(comm, n, n, n)
        mesh = dolfinx.mesh.create_mesh(
            MPI.COMM_WORLD,
            np.array(cube.geometry.dofmap, dtype=np.int64),
            np.array([[p[0], p[1], p[2] - 1.0] for p in cube.geometry.x]),
            ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
        )
        space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        l_space = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 0))
        super().__init__(mesh, space, l_space, coupling_space)

        self.set_bc(lambda x: np.zeros_like(x[0]), lambda x: np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
            | np.isclose(x[2], -1.0),
        )
        self.set_f(rhs_f)
        self.set_a(ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx)
        self.set_f_form(ufl.inner(self.f, self.v) * ufl.dx)
        self.set_b(ufl.inner(self.u, self.mu) * ufl.ds)
        self.create_solver(petsc_options)


square2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 7, 7)
square = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD,
    np.array(square2d.geometry.dofmap, dtype=np.int64),
    square2d.geometry.x,
    ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))),
)
space = dolfinx.fem.functionspace(square, ("Discontinuous Lagrange", 0))

positive = DolfinxPositiveSide(7, space)
negative = DolfinxNegativeSide(7, space)

# Check that each half gives the correct solution
exact_lamb = dolfinx.fem.Function(space)
exact_lamb.interpolate(lambda x: grad_u_exact(x)[2])

exact = dolfinx.fem.Function(space)
exact.interpolate(u_exact)


pos_u_exact = dolfinx.fem.Function(positive.space)
pos_u_exact.interpolate(u_exact)
pos_lamb_exact = dolfinx.fem.Function(positive.lambda_space)
def ff(x):
    return -grad_u_exact(x)[2]
pos_lamb_exact.interpolate(lambda x: np.array([-ff(i) for i in x.T]))

print("<Au-B^tl = f")
lhs = positive.a_mat_scipy @ pos_u_exact.x.array - positive.B_T(pos_lamb_exact).x.array

rhs = dolfinx.fem.assemble_vector(dolfinx.fem.form(positive.f_form))
dolfinx.fem.apply_lifting(rhs.array, [positive.a_form], bcs=[[positive.bc]])
positive.bc.set(rhs.array)
rhs = rhs.array


print("", (positive.a_mat_scipy @ pos_u_exact.x.array)[:5])
print("", positive.B_T(pos_lamb_exact).x.array[:5])
print(lhs[:5])
print(rhs[:5])
print(np.linalg.norm(lhs))
print(np.linalg.norm(rhs))
print(np.linalg.norm(rhs - lhs) / np.linalg.norm(lhs))
print(np.linalg.norm(rhs - lhs) / np.linalg.norm(rhs))
print(">")
print("<Bu=0")
lhs = positive.B(pos_u_exact).x.array
print(np.linalg.norm(lhs))
print(">")


print("<<")
lhs2 = dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.inner(pos_u_exact, positive.v) * ufl.dx))
dolfinx.fem.apply_lifting(lhs2.array, [positive.a_form], bcs=[[positive.bc]])
positive.bc.set(lhs2.array)
lhs2 = lhs2.array
lhs3 = pos_u_exact.x.array

au = dolfinx.fem.Function(positive.space)
au.x.array[:] = positive.a_mat_scipy @ pos_u_exact.x.array
rhs2 = positive.A_inverse(au).x.array

print(lhs2[:5])
print(lhs3[:5])
print(rhs2[:5])
print(pos_u_exact.x.array[:5])
print(np.linalg.norm(lhs2))
print(np.linalg.norm(rhs2))
print(np.linalg.norm(rhs2 - lhs2) / np.linalg.norm(lhs2))
print(np.linalg.norm(rhs2 - lhs2) / np.linalg.norm(rhs2))
print(">>")

print("<<<")
#lhs_form = dolfinx.fem.form(ufl.inner(positive.u, positive.v) * ufl.dx)
#lhs2 = dolfinx.fem.assemble_vector(lhs_form)
#dolfinx.fem.apply_lifting(lhs2.array, [positive.a_form], bcs=[[positive.bc]])
#positive.bc.set(lhs2.array)
#lhs2 = lhs2.array

#rf = dolfinx.fem.Function(positive.space)
#rf.x.array[:] = rhs

#rhs2 = positive.A_inverse(pos_lamb_exact).x.array - positive.A_inverse(positive.f).x.array

#print(min(lhs2), max(lhs2))
#print(min(rhs2), max(rhs2))
#print(np.linalg.norm(lhs2))
#print(np.linalg.norm(rhs2))
#print(np.linalg.norm(rhs2 - lhs2) / np.linalg.norm(lhs2))
#print(np.linalg.norm(rhs2 - lhs2) / np.linalg.norm(rhs2))

print(">>>")


# Solve a problem
ndofs = space.dofmap.index_map.size_global
mu = ufl.TestFunction(space)

rhs = sum(
    -dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.inner(g, mu) * ufl.dx)).array
    for g in [
        positive.trace(positive.B(positive.A_inverse(positive.f))),
        negative.trace(negative.B(negative.A_inverse(negative.f))),
    ]
)

rhs = np.concatenate([
    -dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.inner(g, mu) * ufl.dx)).array
    for g in [
        positive.trace(positive.B(positive.A_inverse(positive.f))),
        negative.trace(negative.B(negative.A_inverse(negative.f))),
    ]
])

def lhs(coeffs):
    pos_l = dolfinx.fem.Function(space)
    pos_l.x.array[:] = coeffs[:ndofs]
    neg_l = dolfinx.fem.Function(space)
    neg_l.x.array[:] = coeffs[ndofs:]

    pos = positive.trace(positive.B(positive.A_inverse(positive.B_T(positive.mollify(pos_l)))))
    neg = negative.trace(negative.B(negative.A_inverse(negative.B_T(negative.mollify(neg_l)))))

    #return dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.inner(pos, mu) * ufl.dx)).array + dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.inner(neg, mu) * ufl.dx)).array
    return np.concatenate([i.array for i in [
        dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.inner(pos, mu) * ufl.dx)),
        dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.inner(neg, mu) * ufl.dx)),
    ]])

from scipy.sparse.linalg import LinearOperator, gmres

S = LinearOperator((ndofs*2, ndofs*2), matvec=lhs)
# S = LinearOperator((ndofs, ndofs), matvec=lhs)

def f(x):
    print("", x)

#print(S @ np.concatenate([exact_lamb.x.array] * 2))
#print("=")
#print(rhs)

err = np.linalg.norm(S @ np.concatenate([exact_lamb.x.array]*2) - rhs) / np.linalg.norm(rhs)
print("err =", err)
assert err < 0.1

sol, info = gmres(S, rhs, callback=f, maxiter=100)
print(info)

lamb = dolfinx.fem.Function(space)
lamb.x.array[:] = sol

print(lamb.x.array)
print(exact_lamb.x.array)

print("lamb_error =", dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(lamb-exact_lamb, lamb-exact_lamb) * ufl.dx)) / dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(lamb, lamb) * ufl.dx)))

#pos_u = positive.A_inverse(positive.B_T(positive.mollify(lamb)) + positive.f)
#pos_u_exact = dolfinx.fem.Function(positive.space)
#pos_u_exact.interpolate(u_exact)
#print("=0")
#print([i for i in positive.B(pos_u_exact).x.array if abs(i) > 1e-8])
#print("=0")
#print(np.abs(pos_u.x.array - pos_u_exact.x.array))
#print(np.linalg.norm(pos_u.x.array - pos_u_exact.x.array))
#print("u_error =", dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(pos_u-pos_u_exact, pos_u-pos_u_exact) * ufl.dx)) / dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(pos_u, pos_u) * ufl.dx)))
# print(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(neg_u-exact, neg_u-exact) * ufl.dx)))


