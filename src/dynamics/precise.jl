# Positions to about 10⁻¹⁵ m in pairs of Float32 (double-float arithmetic: Dekker 1971, Knuth's
# TwoSum, Dekker's TwoProd), so that every device runs it, Metal included, which has no
# Float64. Only correctly rounded sums and products are assumed, not a fused multiply-add.
#
# Float32 resolves a coordinate of 0.05 m to 4 nm. A paper vertex's weight presses it 0.04 nm
# into what it rests on at the stiffness a roller nip needs, so a sheet's contact forces come out
# as noise twenty times its weight, rectified by the contact's one-sidedness into five times its
# weight on a slope; and a stiff membrane turns the same nm into forces of 10⁻⁵ N per vertex.
# Differences of nearby positions and distances to shapes are taken here before anything is
# rounded to Float32.
#
# The error-free transformations only work as written: an optimizer that reassociates float
# arithmetic folds their error terms to zero. Mesa does that to SPIR-V arithmetic not marked
# `NoContraction` (TwoSum's error terms came back 0 on a Radeon 8060S and on lavapipe), which
# Lava now marks wherever Julia does not allow contraction; Metal compiles with fast math by
# default. So every intermediate also passes through a multiplication by `one`, a 1 read from
# device memory at run time, which no compiler can see through. On the host it costs a multiply.

"""
A position (or a velocity) as `hi + lo`, `lo` far below `hi`'s last bit.
"""
struct Position
    hi::Vec3f
    lo::Vec3f
end
Position(p::Vec3f) = Position(p, zero(Vec3f))
Position(p) = (h = Vec3f(p); Position(h, Vec3f(Vec3{Float64}(p) - Vec3{Float64}(h))))

Base.zero(::Type{Position}) = Position(zero(Vec3f), zero(Vec3f))

"The position rounded to Float32."
@inline approximate(p::Position) = p.hi + p.lo

"a + b exactly as s + e (Knuth); `one` hides the arithmetic from optimizers."
@inline function two_sum(a::Float32, b::Float32, one::Float32)
    s = (a + b) * one
    bb = (s - a) * one
    t = (a - (s - bb) * one) * one
    return s, t + (b - bb) * one
end

"a + b exactly as s + e, for |a| ≥ |b|."
@inline function fast_two_sum(a::Float32, b::Float32, one::Float32)
    s = (a + b) * one
    return s, (b - (s - a) * one) * one
end

"Restore the leading/trailing invariant after cancellation between two position pairs."
@inline function renormalized(hi::Vec3f, lo::Vec3f, one)
    x, xl = two_sum(hi[1], lo[1], one)
    y, yl = two_sum(hi[2], lo[2], one)
    z, zl = two_sum(hi[3], lo[3], one)
    return Position(Vec3f(x, y, z), Vec3f(xl, yl, zl))
end

"`a` as hi + lo with 12 significant bits each (Veltkamp), so that their products are exact."
@inline function halves(a::Float32, one::Float32)
    c = (4097.0f0 * a) * one
    h = (c - (c - a) * one) * one
    return h, (a - h) * one
end

"""
a × b exactly as p + e (Dekker), from correctly rounded products and sums alone: Vulkan lets a
driver evaluate `fma` as a product and a sum (lavapipe does), which turns its error term into 0.
"""
@inline function two_prod(a::Float32, b::Float32, one::Float32)
    p = (a * b) * one
    ah, al = halves(a, one)
    bh, bl = halves(b, one)
    e = ((((ah * bh) * one - p) * one + (ah * bl) * one) * one + (al * bh) * one) * one + (al * bl) * one
    return p, e * one
end

"Scalar double-float operations, also used by small dense factorizations on the device."
@inline function pair_add(ah, al, bh, bl, one)
    h, e = two_sum(ah, bh, one)
    return fast_two_sum(h, (e + al + bl) * one, one)
end
@inline function pair_mul(ah, al, bh, bl, one)
    h, e = two_prod(ah, bh, one)
    return fast_two_sum(h, (e + ah * bl + al * bh + al * bl) * one, one)
end
@inline function pair_div(ah, al, bh, bl, one)
    q = ah / bh
    ph, pl = pair_mul(q, 0.0f0, bh, bl, one)
    rh, rl = pair_add(ah, al, -ph, -pl, one)
    return fast_two_sum(q, (rh + rl) / bh, one)
end
@inline function pair_sqrt(ah, al, one)
    q = sqrt(ah)
    ph, pl = two_prod(q, q, one)
    rh, rl = pair_add(ah, al, -ph, -pl, one)
    return fast_two_sum(q, (rh + rl) / (2q), one)
end

@inline function add(p::Position, i, d::Float32, one)
    s, e = two_sum(p.hi[i], d, one)
    return fast_two_sum(s, (e + p.lo[i]) * one, one)
end

"`p` moved by `d`."
@inline function moved(p::Position, d::Vec3f, one)
    x, xl = add(p, 1, d[1], one)
    y, yl = add(p, 2, d[2], one)
    z, zl = add(p, 3, d[3], one)
    return Position(Vec3f(x, y, z), Vec3f(xl, yl, zl))
end

"`p` moved by the pair `d`."
@inline moved(p::Position, d::Position, one) = moved(moved(p, d.hi, one), d.lo, one)

# Optional FP64 elastic Krylov vectors on devices with native FP64. Retain the
# correction's low limb when committing it to the existing position format.
@inline moved(p::Position,d::Vec3d,one)=moved(p,Position(d),one)

"""
`d / t` as a pair, to the pair's precision: the quotient, and the remainder d − q t (exact
through TwoProd) divided again.
"""
@inline function quotient(d::Position, t::Float32, one)
    q = d.hi / t
    return Position(q, Vec3f(remainder(d, q, t, 1, one), remainder(d, q, t, 2, one), remainder(d, q, t, 3, one)))
end

"Component `k` of (d − q t) / t, the remainder taken exactly."
@inline function remainder(d::Position, q::Vec3f, t::Float32, k, one)
    p, e = two_prod(q[k], t, one)
    return (((d.hi[k] - p) * one - e) * one + d.lo[k]) / t
end

"`v t` as a pair: what a velocity carries a point in a step, to the pair's precision."
@inline function scaled(v::Position, t::Float32, one)
    p1, e1 = two_prod(v.hi[1], t, one)
    p2, e2 = two_prod(v.hi[2], t, one)
    p3, e3 = two_prod(v.hi[3], t, one)
    return renormalized(Vec3f(p1, p2, p3), Vec3f(e1, e2, e3) + v.lo * t, one)
end

"""
`p − q` for nearby points, to the precision of their difference: the leading parts subtract
exactly (Sterbenz), so no error term is needed.
"""
@inline Base.:-(p::Position, q::Position) = (p.hi - q.hi) + (p.lo - q.lo)

"""
`p − c` for a far point `c`, as a big part and the rest: `hi` exact differences of the leading
parts, `lo` what they rounded away plus `p.lo`.
"""
@inline function offset(p::Position, c::Vec3f, one)
    x, xe = two_sum(p.hi[1], -c[1], one)
    y, ye = two_sum(p.hi[2], -c[2], one)
    z, ze = two_sum(p.hi[3], -c[3], one)
    return renormalized(Vec3f(x, y, z), Vec3f(xe, ye, ze) + p.lo, one)
end

"`n · q` as an unrounded pair (hi, lo)."
@inline function dot_pair(n::Vec3f, q::Position, one)
    s, e = two_prod(n[1], q.hi[1], one)
    p2, e2 = two_prod(n[2], q.hi[2], one)
    s, f = two_sum(s, p2, one)
    e = (e + e2 + f) * one
    p3, e3 = two_prod(n[3], q.hi[3], one)
    s, f = two_sum(s, p3, one)
    return s, (e + e3 + f + dot(n, q.lo)) * one
end

"""
`|n|² − 1` for a Float32 unit vector, to full precision. Normalized in Float32, a normal is
unit only to 10⁻⁷, and a distance of 0.1 mm taken along it comes out 6 pm off, 15 % of what a
sheet's weight presses it in: `(d / |n|)` is `d (1 − excess / 2)`.
"""
@inline function unit_excess(n::Vec3f, one)
    a, ea = two_prod(n[1], n[1], one)
    b, eb = two_prod(n[2], n[2], one)
    c, ec = two_prod(n[3], n[3], one)
    s, f = two_sum(a, b, one)
    s, g = two_sum(s, c, one)
    s, h = two_sum(s, -1.0f0, one)
    return s + (f + g + h + ea + eb + ec) * one
end

"`(a² + b²) − c²` for a = ah + al etc., to the precision of the result: what a distance from an axis needs."
@inline function square_difference(ah, al, bh, bl, c::Float32, one)
    a2, ea = two_prod(ah, ah, one)
    b2, eb = two_prod(bh, bh, one)
    c2, ec = two_prod(c, c, one)
    s, f = two_sum(a2, b2, one)
    s, g = two_sum(s, -c2, one)
    return s + (f + g + ea + eb - ec + 2 * (ah * al + bh * bl)) * one
end

"""
`p − q` as a pair (hi + lo) to full precision, for points a sheet's thickness apart: their
leading parts are not within a factor of two, so their difference rounds (to 7 pm at 0.1 mm).
"""
@inline function difference(p::Position, q::Position, one)
    x, xe = two_sum(p.hi[1], -q.hi[1], one)
    y, ye = two_sum(p.hi[2], -q.hi[2], one)
    z, ze = two_sum(p.hi[3], -q.hi[3], one)
    return renormalized(Vec3f(x, y, z), Vec3f(xe, ye, ze) + (p.lo - q.lo), one)
end

"`|d|²` for `d` = hi + lo as an unrounded pair (hi, lo)."
@inline function norm2_pair(d::Position, one)
    d = renormalized(d.hi, d.lo, one)
    a, ea = two_prod(d.hi[1], d.hi[1], one)
    b, eb = two_prod(d.hi[2], d.hi[2], one)
    c, ec = two_prod(d.hi[3], d.hi[3], one)
    s, f = two_sum(a, b, one)
    s, g = two_sum(s, c, one)
    return s, (f + g + ea + eb + ec + 2 * dot(d.hi, d.lo)) * one
end

"""
`|d| − r` for a pair `d` and a length `r` = (hi, lo) it is close to, to the precision of the
result: what a spring's stretch needs, 10⁻⁸ of its length where Float32 resolves 10⁻⁷.
"""
@inline function stretch(d::Position, r::NTuple{2, Float32}, one)
    d = renormalized(d.hi, d.lo, one)
    denominator = norm(d.hi) + r[1]
    denominator == 0 && return norm(d.lo) - r[2]
    s, e = norm2_pair(d, one)
    rr, f = two_prod(r[1], r[1], one)
    q, g = two_sum(s, -rr, one)
    return (q + (g + e - f - 2 * r[1] * r[2]) * one) / denominator
end

"`|d| − |c|` for two pairs of about the same length, to the precision of the result."
@inline function stretch(d::Position, c::Position, one)
    d = renormalized(d.hi, d.lo, one)
    c = renormalized(c.hi, c.lo, one)
    denominator = norm(d.hi) + norm(c.hi)
    denominator == 0 && return norm(d.lo) - norm(c.lo)
    s, e = norm2_pair(d, one)
    r, f = norm2_pair(c, one)
    q, g = two_sum(s, -r, one)
    return (q + (g + e - f) * one) / denominator
end

"`p − q` for two pairs, as a pair."
@inline minus(p::Position, q::Position, one) = difference(p, q, one)

"`|d|` of a pair, to its precision."
@inline function length_pair(d::Position, one)
    d = renormalized(d.hi, d.lo, one)
    s, e = norm2_pair(d, one)
    r = sqrt(s)
    return r + e / (2 * max(r, 1.0f-30))
end

"""
Point `p` against segment `a`–`b` (pairs): the parameter of the closest point along it, and
the offset from that point to `p` as a pair.
"""
@inline function segment_offset(p::Position, a::Position, b::Position, one)
    e = b - a
    t = clamp(dot(p - a, e) / max(dot(e, e), 1.0f-30), 0.0f0, 1.0f0)
    return t, minus(difference(p, a, one), scaled(difference(b, a, one), t, one), one)
end
