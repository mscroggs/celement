import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

# Problem with exact solution u = sin(pi*z)x(x-1)y(y-1)
def exact(x):
    return np.sin(np.pi*x[2])*x[0]*(x[0]-1)*x[1]*(x[1]-1)


def grad_u_exact(x):
    return np.array([
        np.sin(np.pi * x[2]) * x[1] * (x[1] - 1) * (2 * x[0] - 1),
        np.sin(np.pi * x[2]) * x[0] * (x[0] - 1) * (2 * x[1] - 1),
        np.pi * np.cos(np.pi * x[2]) * x[0] * (x[0] - 1) * x[1] * (x[1] - 1)
    ])


def rhs_f(x):
    return -np.sin(np.pi * x[2]) * (x[1] * (x[1] - 1) * 2 + x[0] * (x[0] - 1) * 2 - np.pi**2 * x[0] * (x[0] - 1) * x[1] * (x[1] - 1))


n = 10
mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
l_space = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 0))

diri_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim=2, marker=lambda x: np.isclose(x[0], 0.0)
    | np.isclose(x[0], 1.0)
    | np.isclose(x[1], 0.0)
    | np.isclose(x[1], 1.0)
    | np.isclose(x[2], 1.0),
)
diri_dofs = dolfinx.fem.locate_dofs_topological(V=space, entity_dim=2, entities=diri_facets)
bcf = dolfinx.fem.Function(space)
bcf.interpolate(exact)
bc = dolfinx.fem.dirichletbc(value=bcf, dofs=np.array(diri_dofs, dtype=np.int32))

u = ufl.TrialFunction(space)
v = ufl.TestFunction(space)

lamb = ufl.TrialFunction(l_space)

a_form = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
a_mat = dolfinx.fem.assemble_matrix(a_form, bcs=[bc]).to_scipy()

b_form = dolfinx.fem.form(ufl.inner(lamb, v) * ufl.ds)
b_mat = dolfinx.fem.assemble_matrix(b_form, bcs=[bc]).to_scipy()

f = dolfinx.fem.Function(space)
f.interpolate(rhs_f)
f_form = dolfinx.fem.form(ufl.inner(f, v) * ufl.dx)
f_vec = dolfinx.fem.assemble_vector(f_form)
dolfinx.fem.apply_lifting(f_vec.array, [a_form], bcs=[[bc]])
bc.set(f_vec.array)

u = dolfinx.fem.Function(space)
u.interpolate(exact)

du_dn = dolfinx.fem.Function(l_space)
du_dn.interpolate(lambda x: -grad_u_exact(x)[2])

print("<Au,v> - <lamb,v> = <f,v>")
lhs = a_mat @ u.x.array - b_mat @ du_dn.x.array
rhs = f_vec.array
print(lhs)
print(rhs)
print(min(rhs), max(rhs))
print(min(lhs), max(lhs))
print(np.linalg.norm(rhs - lhs))
print(np.linalg.norm(rhs - lhs) / np.linalg.norm(lhs))
print(np.linalg.norm(rhs - lhs) / np.linalg.norm(rhs))
print(">")
