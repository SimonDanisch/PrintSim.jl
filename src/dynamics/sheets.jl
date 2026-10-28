# Elastic forces of a sheet on its vertices and the blocks of their Hessian: the diagonal block of
# a vertex and the block that couples it to the next vertex on its grid line, which is what the
# line solve in `lines.jl` needs.
#
# In-plane: springs along the grid's edges and diagonals. Paper strains by 10⁻⁴ at most, so its
# constitutive law in the plane does not matter, only that it is stiff and isotropic enough;
# springs give positive semi-definite blocks between any two vertices, which the line solve needs
# (Newton's Neo-Hookean membrane only gives the diagonal ones). A spring has geometric stiffness
# across its direction only under tension: a sheet under no stress offers none out of its plane.
#
# Bending: the dihedral angle of every interior edge (Newton's `particle_vbd_kernels.py`),
# weighted by |e|²/(A₁ + A₂) so the energy does not depend on the grid spacing (Grinspun et al.
# 2003), with a factor calibrated per grid in `add_sheet!`. Gauss-Newton blocks k ∇θₐ ∇θ_bᵀ.

const I3 = Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1)

"Scaled SPD block inverse; a poorly resolved block uses positive diagonal preconditioning."
@inline function preconditioner_inverse(D::Mat{3,3,T,9}) where {T<:AbstractFloat}
    scale = max(abs(D[1, 1]), abs(D[2, 2]), abs(D[3, 3]), T(1.0f-30))
    A = D / scale
    determinant = det(A)
    floor=T===Float32 ? T(1.0f-12) : 16eps(T)
    if isfinite(determinant) && determinant > floor
        inverse = inv(A) / scale
        all(isfinite, inverse) && return inverse
    end
    return Mat{3,3,T,9}(1 / max(D[1, 1], T(1.0f-30)), 0, 0,
        0, 1 / max(D[2, 2], T(1.0f-30)), 0,
        0, 0, 1 / max(D[3, 3], T(1.0f-30)))
end

"""
Spring `e` on its vertex `order` (0 or 1): force on that vertex, its diagonal block, and the
block coupling it to the other end.
"""
@inline function spring(m, x, anchor, e, order, dt)
    @inbounds begin
        a, b = m.springs[e]
        one = m.one[1]
        d = difference(x[b], x[a], one)
        l = norm(d.hi)
        n = d.hi / l
        k = m.spring_stiffness[e]
        t = k * stretch(d, m.spring_rest[e], one)
        kd = m.spring_damping[e]
        if kd > 0
            t += kd * k * stretch(d, difference(anchor[b], anchor[a], one), one)
        end
        Hd = (k * (1 + kd)) * (n * n') + (max(t, 0.0f0) / l) * (I3 - n * n')
        f = order == 0 ? t * n : -t * n
        return f, Hd, -Hd
    end
end

"Derivative of a normalized vector, given the derivative of the unnormalized one."
@inline normalized_derivative(len, n, d) = (I3 - n * n') * d / len

@inline skew(v) = Mat3f(0, v[3], -v[2], -v[3], 0, v[1], v[2], -v[1], 0)

@inline function angle_derivative(n1, n2, e, dn1, dn2, s, c, s1, s2)
    ds = (s1 * dn2 - s2 * dn1)' * e
    dc = dn1' * n2 + dn2' * n1
    return ds * c - dc * s
end

"Dihedral angle of a hinge with corners x0..x3, as `dihedral` measures it."
@inline function hinge_angle(x0, x1, x2, x3)
    n1 = normalize(cross(x2 - x0, x3 - x0))
    n2 = normalize(cross(x3 - x1, x2 - x1))
    return atan(dot(cross(n1, n2), normalize(x3 - x2)), dot(n1, n2))
end

"""
Angle of a hinge with corners x0..x3 and its gradient with respect to each corner. All four
at once, selected by masks afterwards like Newton does: a branch per corner becomes a switch
on an i63 that Lava cannot compile.
"""
@inline function hinge_gradients(x0, x1, x2, x3)
    x02, x03, x13, x12, e = x2 - x0, x3 - x0, x3 - x1, x2 - x1, x3 - x2
    n1 = cross(x02, x03)
    n2 = cross(x13, x12)
    l1, l2, le = norm(n1), norm(n2), norm(e)
    n1, n2, eh = n1 / l1, n2 / l2, e / le
    s = dot(cross(n1, n2), eh)
    c = dot(n1, n2)
    s1, s2 = skew(n1), skew(n2)
    z = zero(Mat3f)
    g0 = angle_derivative(n1, n2, eh, normalized_derivative(l1, n1, skew(e)), z, s, c, s1, s2)
    g1 = angle_derivative(n1, n2, eh, z, normalized_derivative(l2, n2, -skew(e)), s, c, s1, s2)
    g2 = angle_derivative(n1, n2, eh, normalized_derivative(l1, n1, -skew(x03)), normalized_derivative(l2, n2, skew(x13)), s, c, s1, s2)
    g3 = angle_derivative(n1, n2, eh, normalized_derivative(l1, n1, skew(x02)), normalized_derivative(l2, n2, -skew(x12)), s, c, s1, s2)
    return atan(s, c), g0, g1, g2, g3
end

@inline pick(g0, g1, g2, g3, k) = Float32(k == 0) * g0 + Float32(k == 1) * g1 + Float32(k == 2) * g2 + Float32(k == 3) * g3

"""
Hinge `h` on its vertex `order` (0: opposite 1, 1: opposite 2, 2: edge start, 3: edge end):
force on that vertex, its diagonal block, and the block coupling it to vertex `other` of the
hinge (−1 for none). Energy ½ k (θ − θ₀)².
"""
@inline function bending(m, x, anchor, h, order, other, dt)
    @inbounds begin
        i0, i1, i2, i3 = m.hinges[h]
        o = x[i0]
        θ, g0, g1, g2, g3 = hinge_gradients(zero(Vec3f), x[i1] - o, x[i2] - o, x[i3] - o)
        dθ = pick(g0, g1, g2, g3, order)
        dθo = pick(g0, g1, g2, g3, other)
        k = m.hinge_stiffness[h]
        kd = m.hinge_damping[h]
        moment = k * (θ - m.hinge_rest[h])
        if kd > 0
            a = anchor[i0]
            moment += kd * k * (θ - hinge_angle(zero(Vec3f), anchor[i1] - a, anchor[i2] - a, anchor[i3] - a))
        end
        kk = k * (1 + kd)
        return -moment * dθ, kk * (dθ * dθ'), kk * (dθ * dθo')
    end
end
