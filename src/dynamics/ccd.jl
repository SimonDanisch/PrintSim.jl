# Tight Inclusion parameter-box CCD, adapted from:
# https://github.com/Continuous-Collision-Detection/Tight-Inclusion
# revision 6f84001a790a9d3362b7c13d410037140d83fa2c
# MIT License
# Copyright (c) 2020 Bolun Wang, Zachary Ferguson, Teseo Schneider,
# Xin Jiang, Marco Attene, and Daniele Panozzo
# The filtered orientation helpers also adapt ExactPredicates.jl's
# generated accurate floating-point filter: Copyright (c) 2019 Pierre Lairez.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The inclusion functions are multi-affine in (t,u,v); their eight corner
# evaluations enclose every value on a parameter box. Vertex/face boxes outside
# u+v<=1 are discarded. Pending boxes are visited in chronological order.
# Changes for device use: fixed-capacity queue, dyadic floating parameters,
# generic Float32/Float64 vectors, and explicit unresolved results on every
# resource/numerical limit. An unresolved query NEVER reports separation.
# Full assembly candidate generation and step integration are separate from
# these narrow-phase primitives.

"ExactPredicates.jl's certified Float64 fast filter; 2 means inconclusive, NOT collinear."
@inline function ccd_orient2_filter(p, q, r)
    u1, u2 = p[1] - r[1], p[2] - r[2]
    v1, v2 = q[1] - r[1], q[2] - r[2]
    determinant = u1 * v2 - u2 * v1
    # ExactPredicates.Codegen.accumulator + fastfilter for orient, with each
    # difference vector grouped separately. Omit exponent/zero shortcuts and
    # CPU interval/rational fallbacks; uncertain device queries keep refining.
    error = max(abs(u1), abs(u2)) * max(abs(v1), abs(v2)) * 8.881784197001262e-16
    if !issubnormal(error) && error > 0 && isfinite(determinant) && abs(determinant) > error
        return determinant > 0 ? Int32(1) : Int32(-1)
    end
    return Int32(2)
end

@inline function ccd_orient2_margin(p, q, r, padding)
    sign = ccd_orient2_filter(p, q, r)
    sign == 2 && return sign
    edge = (q[1] - p[1], q[2] - p[2])
    determinant = edge[1] * (r[2] - p[2]) - edge[2] * (r[1] - p[1])
    # Retain the inclusion filter's coordinate-roundoff allowance as well as
    # the predicate's arithmetic certificate. A merely tiny sign is not a gap.
    abs(determinant) > 2padding * (abs(edge[1]) + abs(edge[2])) || return Int32(2)
    return sign
end

@inline function ccd_same_strict_side(a, b, c, d, padding)
    sign = ccd_orient2_margin(a, b, c, padding)
    return sign != 2 && sign == ccd_orient2_margin(a, b, d, padding)
end

"ExactPredicates.jl's orient3 accurate filter, with an additional distance margin."
@inline function ccd_spatial_static_separated(x0, x1, kind, padding)
    return false
end
@inline function ccd_spatial_static_separated(x0, x1, ::Val{VF}, padding::Float64) where {VF}
    @inbounds for k in Int32(1):Int32(4)
        x0[k] == x1[k] || return false
    end
    p, q, r, a = x0
    u, v, w = p - a, q - a, r - a
    # Keep the upstream determinant operation order and grouped error bound.
    determinant = (u[1] * (v[2] * w[3] - v[3] * w[2]) -
        v[1] * (u[2] * w[3] - u[3] * w[2])) +
        w[1] * (u[2] * v[3] - u[3] * v[2])
    error = maximum(abs, u) * maximum(abs, v) * maximum(abs, w) * 4.884981308350698e-15
    # Noncoplanarity separates a point from its triangle plane, or two edges
    # from each other's supporting lines. Bound the normal's L1 norm without
    # cancellation; factor 12 (rather than 6) also covers rounded differences.
    edge1 = VF ? r - q : q - p
    edge2 = VF ? a - q : a - r
    normal_bound = 12maximum(abs, edge1) * maximum(abs, edge2)
    margin = error + padding * normal_bound
    return !issubnormal(error) && error > 0 && isfinite(margin) &&
        isfinite(determinant) && abs(determinant) > margin
end

"Separating-line certificates in coordinate projections of stationary zero-thickness primitives."
@inline ccd_planar_static_separated(x0, x1, kind, padding) = false
@inline function ccd_planar_static_separated(x0, x1, ::Val{VF}, padding::Float64) where {VF}
    # Dispatch, rather than a device-side Type comparison, keeps the Metal /
    # coarse Float32 shader completely free of Float64 arithmetic and Type objects.
    @inbounds for k in Int32(1):Int32(4)
        x0[k] == x1[k] || return false
    end
    @inbounds for axis in Int32(1):Int32(3)
        # A strict separation in ANY coordinate projection also separates the
        # spatial primitives: orthogonal projection cannot increase distance.
        # Coplanarity is not required. Keep the padded distance certificate;
        # overlapping projections remain inconclusive, never imply contact.
        u = axis == 3 ? Int32(1) : axis + Int32(1)
        v = axis == 1 ? Int32(3) : axis == 2 ? Int32(1) : Int32(2)
        p = ntuple(k -> (x0[k][u], x0[k][v]), Val(4))
        if VF
            orientation = ccd_orient2_filter(p[2], p[3], p[4])
            orientation == 2 && continue
            (ccd_orient2_margin(p[2], p[3], p[1], padding) == -orientation ||
                ccd_orient2_margin(p[3], p[4], p[1], padding) == -orientation ||
                ccd_orient2_margin(p[4], p[2], p[1], padding) == -orientation) && return true
        else
            (ccd_same_strict_side(p[1], p[2], p[3], p[4], padding) ||
                ccd_same_strict_side(p[3], p[4], p[1], p[2], padding)) && return true
        end
    end
    return false
end

@inline function ccd_value(x0, x1, t, u, v, ::Val{VF}) where {VF}
    a = x0[1] + t * (x1[1] - x0[1])
    b = x0[2] + t * (x1[2] - x0[2])
    c = x0[3] + t * (x1[3] - x0[3])
    d = x0[4] + t * (x1[4] - x0[4])
    return VF ? a - ((c - b) * u + (d - b) * v + b) :
        ((b - a) * u + a) - ((d - c) * v + c)
end

@inline function ccd_box(x0, x1, box, kind, padding)
    first = ccd_value(x0, x1, box[1], box[3], box[5], kind)
    lo, hi = first, first
    for corner in Int32(1):Int32(7)
        t = box[1 + ((corner >> 2) & 1)]
        u = box[3 + ((corner >> 1) & 1)]
        v = box[5 + (corner & 1)]
        value = ccd_value(x0, x1, t, u, v, kind)
        lo, hi = min.(lo, value), max.(hi, value)
    end
    # Minimum separation is Euclidean, not a cube with that half-width.
    # Distance from the origin to this enclosing AABB is a lower bound for
    # every primitive value in the parameter box. The farthest AABB corner
    # supplies the corresponding upper bound. This also avoids accepting a
    # diagonal gap inside an L-infinity cube as a resolved thickness contact.
    nearest = max.(lo, -hi, zero(eltype(lo)))
    farthest = max.(abs.(lo), abs.(hi))
    radius2 = padding * padding
    overlap = sum(abs2, nearest) <= radius2
    inside = sum(abs2, farthest) <= radius2
    return overlap, maximum(hi - lo), inside
end

"""
    tight_ccd(x0, x1, Val(vertex_face), separation, tolerance, max_iterations)

Four primitive vertices at each end of a linear trajectory. For vertex/face,
vertex 1 is the point and 2:4 are the triangle; otherwise 1:2 and 3:4 are edges.
Returns `(hit, time, resolved)`. `time` is a lower bound on possible impact;
`resolved=false` means the budget/precision prevented completing the query.
Callers must reduce their step or reject it on an unresolved result.
"""
function tight_ccd(x0::NTuple{4, V}, x1::NTuple{4, V}, kind::Val{VF},
        separation::T, tolerance::T, max_iterations::Int32) where {V, T, VF}
    z, o = zero(T), one(T)
    if !(isfinite(separation) && isfinite(tolerance) && separation >= z && tolerance > z && max_iterations > 0)
        return (; hit = true, time = z, resolved = false)
    end
    # Roundoff allowance follows the inclusion filter's role, enlarged from
    # the upstream operation-count constants. Do not use @fastmath here.
    scale = z
    for k in Int32(1):Int32(4)
        all(isfinite, x0[k]) && all(isfinite, x1[k]) ||
            return (; hit = true, time = z, resolved = false)
        scale = max(scale, maximum(abs, x0[k]), maximum(abs, x1[k]))
    end
    # Convert the componentwise roundoff filter to a Euclidean radius. The
    # additional relative margin covers rounding in the squared-norm test.
    padding = (separation + sqrt(T(3)) * T(64) * eps(T) * max(scale, floatmin(T))) *
        (o + T(8) * eps(T))
    separation == z && (ccd_spatial_static_separated(x0, x1, kind, padding) ||
        ccd_planar_static_separated(x0, x1, kind, padding)) &&
        return (; hit = false, time = o, resolved = true)
    # These Lipschitz quantities depend on the trajectory, not the current
    # parameter box. Compute them once per pair, outside subdivision.
    speed, usize, vsize = z, z, z
    for k in Int32(0):Int32(7)
        t, u, v = T((k >> 2) & 1), T((k >> 1) & 1), T(k & 1)
        f = ccd_value(x0, x1, t, u, v, kind)
        speed = max(speed, maximum(abs, ccd_value(x0, x1, o - t, u, v, kind) - f))
        usize = max(usize, maximum(abs, ccd_value(x0, x1, t, o - u, v, kind) - f))
        vsize = max(vsize, maximum(abs, ccd_value(x0, x1, t, u, o - v, kind) - f))
    end
    # A Lipschitz safe-time bound before spatial subdivision is essential for
    # parallel overlapping edges: their diagonal parameter strip can exhaust
    # the queue while its time interval still starts at zero. For every spatial
    # parameter, f(t) remains within speed*t of f(0). The initial interval's
    # distance from the padded origin therefore certifies a collision-free
    # prefix without resolving the spatial strip. This is conservative
    # advancement of an inclusion bound, not an assumption that edges diverge.
    initial=ccd_value(x0,x1,z,z,z,kind)
    initial_lo,initial_hi=initial,initial
    for corner in Int32(1):Int32(3)
        value=ccd_value(x0,x1,z,T((corner>>1)&1),T(corner&1),kind)
        initial_lo,initial_hi=min.(initial_lo,value),max.(initial_hi,value)
    end
    gap=max(z,maximum(max.(initial_lo.-padding,-padding.-initial_hi)))
    isfinite(gap) && isfinite(speed) || return (;hit=true,time=z,resolved=false)
    # Round down with an explicit margin; do not start beyond a proven prefix.
    safe_time=speed>z ? max(z,gap/speed-T(16)*eps(T)) : z
    safe_time>o && return (;hit=false,time=o,resolved=true)
    queue = MVector{64, NTuple{6, T}}(undef)
    queue[1] = (safe_time, o, z, o, z, o)
    count = Int32(1)
    for iteration in Int32(1):max_iterations
        count == 0 && return (; hit = false, time = o, resolved = true)
        # Earliest time first; ties use the last inserted box (depth first in
        # the spatial parameters), as in upstream bucket DFS.
        at = count
        @inbounds for k in Int32(1):(count - Int32(1))
            queue[k][1] < queue[at][1] && (at = k)
        end
        @inbounds box = queue[at]
        @inbounds queue[at] = queue[count]
        count -= Int32(1)
        VF && box[3] + box[5] > o && continue
        overlap, width, inside = ccd_box(x0, x1, box, kind, padding)
        if !isfinite(width)
            return (; hit = true, time = box[1], resolved = false)
        end
        overlap || continue
        (width <= tolerance || inside) && return (; hit = true, time = box[1], resolved = true)
        # Split the parameter with the largest geometric variation. Edge
        # lengths and trajectory speed are the same Lipschitz quantities used
        # to derive upstream parameter tolerances.
        tspan, uspan, vspan = box[2] - box[1], box[4] - box[3], box[6] - box[5]
        ts, us, vs = tspan * speed, uspan * usize, vspan * vsize
        axis = ts >= us && ts >= vs ? Int32(1) : us >= vs ? Int32(3) : Int32(5)
        mid = (box[axis] + box[axis + Int32(1)]) / T(2)
        if count >= 63 || !(box[axis] < mid < box[axis + Int32(1)])
            return (; hit = true, time = box[1], resolved = false)
        end
        # Parameter intervals start at [0,1] and use bisection only, so their
        # endpoints remain exactly representable until stagnation above.
        low = ntuple(k -> k == axis + 1 ? mid : box[k], Val(6))
        high = ntuple(k -> k == axis ? mid : box[k], Val(6))
        @inbounds queue[count + 1], queue[count + 2] = high, low
        count += Int32(2)
    end
    earliest = o
    @inbounds for k in Int32(1):count
        earliest = min(earliest, queue[k][1])
    end
    return count == 0 ? (; hit = false, time = o, resolved = true) :
        (; hit = true, time = earliest, resolved = false)
end
