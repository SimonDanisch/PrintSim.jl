# Additive CCD (Li et al. 2021, Codimensional Incremental Potential Contact, Algorithm 1). It
# advances along the step by the current distance over a bound on how fast the distance can
# shrink, and needs no parameter subdivision, so primitive shape does not matter. Tight
# Inclusion's 64-box queue overflows on a 43 mm sliver approaching a shaft 65 µm away and only
# returns a conservative time; repeated, that time shrinks to zero although nothing touches.
# `swept_primitive` therefore settles every query Tight Inclusion leaves unresolved with ACCD.

"Squared distance from `p` to triangle `abc` (Ericson, Real-Time Collision Detection, 5.1.5)."
@inline function point_triangle_distance2(p, a, b, c)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = dot(ab, ap), dot(ac, ap)
    (d1 <= 0 && d2 <= 0) && return sum(abs2, ap)
    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    (d3 >= 0 && d4 <= d3) && return sum(abs2, bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0
        return sum(abs2, ap - (d1 / (d1 - d3)) * ab)
    end
    cp = p - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    (d6 >= 0 && d5 <= d6) && return sum(abs2, cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0
        return sum(abs2, ap - (d2 / (d2 - d6)) * ac)
    end
    va = d3 * d6 - d5 * d4
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return sum(abs2, bp - w * (c - b))
    end
    denominator = va + vb + vc
    v, w = vb / denominator, vc / denominator
    return sum(abs2, ap - v * ab - w * ac)
end

"Squared distance between segments pq and rs (Ericson 5.1.9)."
@inline function segment_segment_distance2(p, q, r, s)
    d1, d2, w = q - p, s - r, p - r
    a, e, f = dot(d1, d1), dot(d2, d2), dot(d2, w)
    z, o = zero(a), one(a)
    (a <= floatmin(a) && e <= floatmin(a)) && return sum(abs2, w)
    a <= floatmin(a) && return sum(abs2, w - clamp(f / e, z, o) * d2)
    c = dot(d1, w)
    e <= floatmin(a) && return sum(abs2, w + clamp(-c / a, z, o) * d1)
    b = dot(d1, d2)
    denominator = a * e - b * b
    sN = denominator > eps(a) * a * e ? clamp((b * f - c * e) / denominator, z, o) : z
    t = (b * sN + f) / e
    if t < 0
        t, sN = z, clamp(-c / a, z, o)
    elseif t > 1
        t, sN = o, clamp((b - c) / a, z, o)
    end
    return sum(abs2, w + sN * d1 - t * d2)
end

"Displacements of the four vertices relative to their mean: the motion that changes distances."
@inline function relative_motion(x0, x1)
    d = ntuple(k -> x1[k] - x0[k], Val(4))
    mean = (d[1] + d[2] + d[3] + d[4]) / 4
    return ntuple(k -> d[k] - mean, Val(4))
end

@inline advance(x, p, step) = ntuple(k -> x[k] + step * p[k], Val(4))

@inline primitive_distance2(x, ::Val{true}) = point_triangle_distance2(x[1], x[2], x[3], x[4])
@inline primitive_distance2(x, ::Val{false}) = segment_segment_distance2(x[1], x[2], x[3], x[4])

"""
    additive_ccd(x0, x1, Val(vertex_face), separation, max_iterations) -> (hit, time, resolved)

C-IPC's additive CCD for one primitive pair moving linearly from `x0` to `x1` (vertex 1 and
triangle 2:4, or edges 1:2 and 3:4). Without contact before the end of the step it returns
`hit = false`; otherwise `time` is a collision-free time at which the gap (beyond `separation`)
has shrunk to a tenth of its initial value. `resolved = false` only if `max_iterations` ran out.
"""
@inline function additive_ccd(x0::NTuple{4, V}, x1::NTuple{4, V}, kind::Val{VF}, separation::T,
        max_iterations::Int32) where {V, T, VF}
    z, o = zero(T), one(T)
    shrink = T(0.1)
    p = relative_motion(x0, x1)
    lp = VF ? norm(p[1]) + max(norm(p[2]), norm(p[3]), norm(p[4])) :
        max(norm(p[1]), norm(p[2])) + max(norm(p[3]), norm(p[4]))
    lp > z || return (; hit = false, time = o, resolved = true)
    x = x0
    d2 = primitive_distance2(x, kind)
    d = sqrt(d2)
    d > separation || return (; hit = true, time = z, resolved = true)
    gap = (d2 - separation^2) / (d + separation)
    target = shrink * gap
    t = z
    step = (o - shrink) * gap / lp
    for _ in Int32(1):max_iterations
        x = advance(x, p, step)
        d2 = primitive_distance2(x, kind)
        d = sqrt(d2)
        gap = (d2 - separation^2) / (d + separation)
        t > z && gap < target && return (; hit = true, time = t, resolved = true)
        t += step
        t > o && return (; hit = false, time = o, resolved = true)
        step = T(0.9) * gap / lp
    end
    return (; hit = true, time = t, resolved = false)
end
