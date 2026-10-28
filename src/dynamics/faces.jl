# Meshes pressing into sheets' faces. A sheet's vertices find the mesh triangles under them
# (`contacts.jl`); between its vertices a sheet is flat, and what pokes into that flat face are
# a mesh's convex corners and edges (`features`): a roller a sheet wraps around (a sheet 6 mm
# between vertices around a 12 mm roller cuts 0.4 mm into it), a guide's edge narrower than the
# sheet's grid. Each sheet triangle keeps up to `FACE_SLOTS` such contacts, a mesh corner
# against the triangle or a mesh edge against one of the triangle's own edges (every edge of a
# sheet belongs to one triangle). The force acts at the contact point on the sheet, shared by
# the triangle's corners at their weights; the corners gather it into the step with the
# stiffness between them, which the multigrid's Galerkin products then carry down. A sheet has
# two sides: each contact remembers on which the feature lies, so one pressed through the sheet
# is pushed back, not on.

"""
Contacts between mesh features and one sheet triangle, at most: a sheet edge 3 mm long crosses
three edges of a 12 mm roller meshed in 32 facets.
"""
const FACE_SLOTS = 4

"""
Each sheet triangle's contacts with mesh features, up to `FACE_SLOTS`: what touches (`kind` 0
none, 1 a mesh corner, 2 a mesh edge), which (`feature`: the mesh vertex, or the mesh edge as
3 (triangle − 1) + edge), for an edge the sheet triangle's edge it touches, on which side of the
sheet, the contact point's weights on the triangle's corners and the normal from the feature to
the sheet, the force and stiffness linearized where the sheet stands, and what friction carries
from step to step.
"""
struct FaceTouches{U, I, F, V, M}
    kind::U
    feature::I
    edge::U
    side::F
    weights::V
    normal::V
    force::V
    block::M
    lagged::F
    spring::V
    sliding::U
end
Adapt.@adapt_structure FaceTouches

function FaceTouches(backend, triangles)
    z(T) = KI.zeros(backend, T, FACE_SLOTS, max(triangles, 1))
    return FaceTouches(z(UInt8), z(Int32), z(UInt8), z(Float32), z(Vec3f), z(Vec3f), z(Vec3f), z(Mat3f), z(Float32), z(Vec3f), z(UInt8))
end

"`p + q` for two pairs, as a pair."
@inline function plus(p::Position, q::Position, one)
    x, xe = two_sum(p.hi[1], q.hi[1], one)
    y, ye = two_sum(p.hi[2], q.hi[2], one)
    z, ze = two_sum(p.hi[3], q.hi[3], one)
    return renormalized(Vec3f(x, y, z), Vec3f(xe, ye, ze) + (p.lo + q.lo), one)
end

"""
Parameters `s`, `t` of the closest points of segments a₀ + s (a₁ − a₀) and b₀ + t (b₁ − b₀)
(Float32, relative to anything), and their squared distance.
"""
@inline function closest_parameters(a0::Vec3f, a1::Vec3f, b0::Vec3f, b1::Vec3f)
    e, f, w = a1 - a0, b1 - b0, a0 - b0
    ee, ff, ef = dot(e, e), dot(f, f), dot(e, f)
    we, wf = dot(w, e), dot(w, f)
    den = ee * ff - ef^2
    s = clamp(ifelse(den > 1.0f-12 * ee * ff, (ef * wf - ff * we) / den, 0.0f0), 0.0f0, 1.0f0)
    t = clamp((ef * s + wf) / max(ff, 1.0f-30), 0.0f0, 1.0f0)
    s = clamp((ef * t - we) / max(ee, 1.0f-30), 0.0f0, 1.0f0)
    d = w + s * e - t * f
    return s, t, dot(d, d)
end

"Whether the closest points of two edges lie inside both, at parameters `s` and `t`."
@inline inside(s, t) = (0 < s < 1) & (0 < t < 1)

"""
Gap between mesh edge `e0`–`e1` and sheet edge `b0`–`b1` (pairs), less `r`, with the feature
remembered on side `side` of the sheet (face normal `face`): the normal from the feature to the
sheet, the sheet point's parameter along its edge, and whether both closest points lie inside
their edges. Pressed through the sheet, the gap is negative and the normal points back.
"""
@inline function edge_gap(e0::Position, e1::Position, b0::Position, b1::Position, face::Vec3f, r::Float32, side::Float32, one)
    s, t, _ = closest_parameters(e0 - b0, e1 - b0, zero(Vec3f), b1 - b0)
    d = minus(plus(difference(e0, b0, one), scaled(difference(e1, e0, one), s, one), one), scaled(difference(b1, b0, one), t, one), one)
    l = length_pair(d, one)
    u = -(d.hi + d.lo) / max(l, 1.0f-30)          # from the feature toward the sheet
    crossed = side * dot(d.hi + d.lo, face) < 0
    g = ifelse(crossed, -l - r, stretch(d, (r, 0.0f0), one))
    return g, ifelse(crossed, -u, u), t, inside(s, t)
end

"""
Gap, normal (feature to sheet) and corner weights of the face contact (`kind`, `feature`,
`edge`, `side`) on the sheet triangle with corners `xa`, `xb`, `xc`, and whether the contact is
this triangle's: a mesh corner over the triangle's inside, a mesh edge crossing the sheet edge
between both edges' ends. Where the closest point reaches an end, the contact is a vertex's
(`contacts.jl`) or the next edge's, and counted here too it doubled theirs: a sheet's middle
row resting on a roller's top edge was held by its own contact and by those of the edges
leaving it, which pulled it down while it rocked.
"""
@inline function face_gap(C::Colliders, kind, feature, edge, side, xa::Position, xb::Position, xc::Position, r, one)
    @inbounds if kind == 0x01
        g, n, w = triangle_gap(C.points[feature], xa, xb, xc, r, side, one)
        return g, -n, w, minimum(w) > 0
    else
        mt = (feature - Int32(1)) ÷ Int32(3) + Int32(1)
        k = feature - Int32(3) * (mt - Int32(1))
        f = C.faces[mt]
        e0 = C.points[ifelse(k == 1, f[1], ifelse(k == 2, f[2], f[3]))]
        e1 = C.points[ifelse(k == 1, f[2], ifelse(k == 2, f[3], f[1]))]
        b0 = ifelse(edge == 0x01, xa, ifelse(edge == 0x02, xb, xc))
        b1 = ifelse(edge == 0x01, xb, ifelse(edge == 0x02, xc, xa))
        g, n, t, own = edge_gap(e0, e1, b0, b1, direction(cross(xb - xa, xc - xa)), r, side, one)
        w = ifelse(edge == 0x01, Vec3f(1 - t, t, 0), ifelse(edge == 0x02, Vec3f(0, 1 - t, t), Vec3f(t, 0, 1 - t)))
        return g, n, w, own
    end
end

"Inserts candidate (`kind`, `feature`, `edge`) at rough distance `d` into the `FACE_SLOTS` closest so far."
@inline function keep_face!(ks, fs, es, ds, kind, feature, edge, d)
    @inbounds if d < ds[FACE_SLOTS]
        k = FACE_SLOTS
        while k > 1 && ds[k - 1] > d
            ks[k], fs[k], es[k], ds[k] = ks[k - 1], fs[k - 1], es[k - 1], ds[k - 1]
            k -= 1
        end
        ks[k], fs[k], es[k], ds[k] = kind, feature, edge, d
    end
    return nothing
end

"""
The mesh features near the sheet triangle with corners `xa`, `xb`, `xc` (within `radius` of its
box), from mesh `m`'s hierarchy: its corners against the triangle, its edges against the
triangle's own edges (`owned`), the closest kept. Those that are the triangle's own contacts
(`face_gap`) come first; the others are kept behind them, for a contact whose closest point
moves inside during the step (a sheet rocking on a roller's edge hands its contact from the
edges on one side of a vertex to those on the other).
"""
@inline function near_features!(ks, fs, es, ds, C::Colliders, m, xa::Position, xb::Position, xc::Position, owned, radius)
    @inbounds begin
        qa, qb, qc = approximate(xa), approximate(xb), approximate(xc)
        lo = min.(qa, qb, qc) .- radius
        hi = max.(qa, qb, qc) .+ radius
        stack = MVector{STACK, Int32}(undef)
        top = Int32(1)
        stack[1] = C.root[m]
        while top > Int32(0)
            node = stack[top]
            top -= Int32(1)
            if all(C.lo[node] .<= hi) && all(lo .<= C.hi[node])
                l = C.left[node]
                if l < 0
                    k = -l
                    while k < -l + C.right[node]
                        mt = C.order[k]
                        mask = C.features[mt]
                        if mask != 0x00
                            f = C.faces[mt]
                            c = 1
                            while c <= 3
                                v = ifelse(c == 1, f[1], ifelse(c == 2, f[2], f[3]))
                                if mask & (UInt8(1) << (c - 1)) != 0x00
                                    p = C.points[v]
                                    w, _, _ = projection(xa - p, xb - p, xc - p)
                                    d, _ = rough_distance(xa - p, xb - p, xc - p)
                                    d <= radius && keep_face!(ks, fs, es, ds, 0x01, v, 0x00, d + ifelse(minimum(w) > 0, 0.0f0, radius))
                                end
                                if mask & (UInt8(1) << (c + 2)) != 0x00
                                    w = ifelse(c == 1, f[2], ifelse(c == 2, f[3], f[1]))
                                    e0, e1 = C.points[v], C.points[w]
                                    j = 1
                                    while j <= 3
                                        if owned & (UInt8(1) << (j - 1)) != 0x00
                                            b0 = ifelse(j == 1, xa, ifelse(j == 2, xb, xc))
                                            b1 = ifelse(j == 1, xb, ifelse(j == 2, xc, xa))
                                            se, te, d2 = closest_parameters(e0 - b0, e1 - b0, zero(Vec3f), b1 - b0)
                                            d = sqrt(d2)
                                            d <= radius && keep_face!(ks, fs, es, ds, 0x02, Int32(3) * (mt - Int32(1)) + Int32(c), UInt8(j), d + ifelse(inside(se, te), 0.0f0, radius))
                                        end
                                        j += 1
                                    end
                                end
                                c += 1
                            end
                        end
                        k += Int32(1)
                    end
                else
                    stack[top + Int32(1)] = l
                    stack[top + Int32(2)] = C.right[node]
                    top += Int32(2)
                end
            end
        end
    end
    return nothing
end

"Area a face contact on a sheet triangle stands for: a third of the triangle's, from its corners' areas."
@inline face_area(area, ia, ib, ic) = @inbounds (area[ia] + area[ib] + area[ic]) / 18

"""
Finds each sheet triangle's contacts with mesh features where the step starts, and carries the
side and friction state of those it already had.
"""
function find_face_touches!(x, inertia, anchor, tris, tri_edges, C::Colliders, F::FaceTouches, one, r::Float32, σ::Float32)
    t = KI.get_global_id(Int32).x
    t <= length(tri_edges) || return nothing
    @inbounds begin
        o = one[1]
        ia, ib, ic = tris[t]
        xa, xb, xc = x[ia], x[ib], x[ic]
        move = max(norm(inertia[ia] - xa) + norm(xa - anchor[ia]), norm(inertia[ib] - xb) + norm(xb - anchor[ib]), norm(inertia[ic] - xc) + norm(xc - anchor[ic]))
        radius = r + RANGE * σ + 2 * move + 1.0f-4
        ks = MVector{FACE_SLOTS, UInt8}(ntuple(_ -> 0x00, FACE_SLOTS))
        fs = MVector{FACE_SLOTS, Int32}(ntuple(_ -> Int32(0), FACE_SLOTS))
        es = MVector{FACE_SLOTS, UInt8}(ntuple(_ -> 0x00, FACE_SLOTS))
        ds = MVector{FACE_SLOTS, Float32}(ntuple(_ -> Inf32, FACE_SLOTS))
        m = Int32(1)
        while m <= Int32(C.meshes)
            near_features!(ks, fs, es, ds, C, m, xa, xb, xc, tri_edges[t], radius)
            m += Int32(1)
        end
        old_kind = MVector{FACE_SLOTS, UInt8}(ntuple(k -> F.kind[k, t], FACE_SLOTS))
        old_feature = MVector{FACE_SLOTS, Int32}(ntuple(k -> F.feature[k, t], FACE_SLOTS))
        old_edge = MVector{FACE_SLOTS, UInt8}(ntuple(k -> F.edge[k, t], FACE_SLOTS))
        old_side = MVector{FACE_SLOTS, Float32}(ntuple(k -> F.side[k, t], FACE_SLOTS))
        old_lagged = MVector{FACE_SLOTS, Float32}(ntuple(k -> F.lagged[k, t], FACE_SLOTS))
        old_spring = MVector{FACE_SLOTS, Vec3f}(ntuple(k -> F.spring[k, t], FACE_SLOTS))
        old_sliding = MVector{FACE_SLOTS, UInt8}(ntuple(k -> F.sliding[k, t], FACE_SLOTS))
        k = 1
        while k <= FACE_SLOTS
            kind = ks[k]
            side, λ, ξ, slides = 0.0f0, 0.0f0, zero(Vec3f), 0x00
            j = 1
            while j <= FACE_SLOTS
                if kind != 0x00 && old_kind[j] == kind && old_feature[j] == fs[k] && old_edge[j] == es[k]
                    side, λ, ξ, slides = old_side[j], old_lagged[j], old_spring[j], old_sliding[j]
                end
                j += 1
            end
            if kind != 0x00 && side == 0
                # New: the side the feature lies on, from the triangle's own normal.
                g1, _, _, _ = face_gap(C, kind, fs[k], es[k], 1.0f0, xa, xb, xc, 0.0f0, o)
                side = ifelse(g1 >= 0, 1.0f0, -1.0f0)
            end
            F.kind[k, t] = kind
            F.feature[k, t] = fs[k]
            F.edge[k, t] = es[k]
            F.side[k, t] = side
            F.lagged[k, t] = λ
            F.spring[k, t] = ξ
            F.sliding[k, t] = slides
            k += 1
        end
    end
    return nothing
end

"""
Force on the contact point and its stiffness for every face contact, linearized where the sheet
stands now, as `mesh_forces!` does for vertices.
"""
function face_forces!(x, anchor, tris, C::Colliders, F::FaceTouches, area, one, r::Float32, σ::Float32, tangential::Float32, motion = StaticMeshes())
    t = KI.get_global_id(Int32).x
    t <= size(F.kind, 2) || return nothing
    @inbounds begin
        ia, ib, ic = tris[t]
        xa, xb, xc = x[ia], x[ib], x[ic]
        a = face_area(area, ia, ib, ic)
        k = 1
        while k <= FACE_SLOTS
            f = zero(Vec3f)
            H = zero(Mat3f)
            kind = F.kind[k, t]
            if kind != 0x00
                m = feature_mesh(C, kind, F.feature[k, t])
                g, n, w, own = face_gap(C, kind, F.feature[k, t], F.edge[k, t], F.side[k, t], xa, xb, xc, r, one[1])
                F.weights[k, t] = w
                F.normal[k, t] = n
                if own && g < RANGE * σ
                    λ, κ = normal_force(g, a * C.grip[3, m], σ)
                    Pt = I3 - n * n'
                    u = Pt * (w[1] * (xa - anchor[ia]) + w[2] * (xb - anchor[ib]) + w[3] * (xc - anchor[ic]) -
                        face_motion(motion, C, kind, F.feature[k, t], F.edge[k, t], xa, xb, xc, k, t, one[1]))
                    ft, Ht = friction_force(Pt, F.spring[k, t], F.sliding[k, t], u, F.lagged[k, t], tangential * κ, C.grip[1, m], C.grip[2, m])
                    f = λ * n + ft
                    H = κ * (n * n') + Ht
                end
            end
            F.force[k, t] = f
            F.block[k, t] = H
            k += 1
        end
    end
    return nothing
end

"The mesh a face contact's feature belongs to."
@inline feature_mesh(C::Colliders, kind, feature) =
    @inbounds kind == 0x01 ? C.point_owner[feature] : C.owner[(feature - Int32(1)) ÷ Int32(3) + Int32(1)]

"""
The face contacts added to the fine level, each sheet vertex gathering from the triangles at it:
its share of each contact's force, w Fᵢ, and of its stiffness, w_k w_l B to each corner l (into
the row sum where l is held, as `coupling!` does), the row sum taking w_k B in all (a mesh
stands still: the contact does not sum to zero over the row).
"""
function face_rows!(A, S, contact, nb, held, tris, vtri_start, vtri, vtri_order, F::FaceTouches)
    i = KI.get_global_id(Int32).x
    i <= length(held) || return nothing
    @inbounds if held[i] == 0x00
        f = zero(Vec3f)
        e = vtri_start[i]
        while e < vtri_start[i + Int32(1)]
            t = vtri[e]
            c = vtri_order[e] + Int32(1)
            ia, ib, ic = tris[t]
            k = 1
            while k <= FACE_SLOTS
                if F.kind[k, t] != 0x00
                    w = F.weights[k, t]
                    B = F.block[k, t]
                    wi = ifelse(c == 1, w[1], ifelse(c == 2, w[2], w[3]))
                    f += wi * F.force[k, t]
                    S[1, i] += wi * B
                    A[CENTER, i] += wi^2 * B
                    S[1, i] += corner_coupling!(A, nb, held, i, ia, wi * w[1] * B)
                    S[1, i] += corner_coupling!(A, nb, held, i, ib, wi * w[2] * B)
                    S[1, i] += corner_coupling!(A, nb, held, i, ic, wi * w[3] * B)
                end
                k += 1
            end
            e += Int32(1)
        end
        contact[i] += f
    end
    return nothing
end

"""
Every face contact at the end of a step, measured again where the sheet ended: its normal force,
and whether it sticks or slides (`friction_state`).
"""
function face_friction_update!(x, anchor, tris, C::Colliders, F::FaceTouches, area, one, r::Float32, σ::Float32, tangential::Float32, motion = StaticMeshes())
    t = KI.get_global_id(Int32).x
    t <= size(F.kind, 2) || return nothing
    @inbounds begin
        ia, ib, ic = tris[t]
        xa, xb, xc = x[ia], x[ib], x[ic]
        a = face_area(area, ia, ib, ic)
        k = 1
        while k <= FACE_SLOTS
            kind = F.kind[k, t]
            λ = 0.0f0
            if kind != 0x00
                m = feature_mesh(C, kind, F.feature[k, t])
                g, n, w, own = face_gap(C, kind, F.feature[k, t], F.edge[k, t], F.side[k, t], xa, xb, xc, r, one[1])
                F.normal[k, t] = n
                if own && g < RANGE * σ
                    λ, κ = normal_force(g, a * C.grip[3, m], σ)
                    Pt = I3 - n * n'
                    u = Pt * (w[1] * (xa - anchor[ia]) + w[2] * (xb - anchor[ib]) + w[3] * (xc - anchor[ic]) -
                        face_motion(motion, C, kind, F.feature[k, t], F.edge[k, t], xa, xb, xc, k, t, one[1]))
                    F.spring[k, t], F.sliding[k, t] = friction_state(Pt * F.spring[k, t], F.sliding[k, t], u, λ, tangential * κ, C.grip[1, m], C.grip[2, m])
                else
                    F.spring[k, t] = zero(Vec3f)
                    F.sliding[k, t] = 0x00
                end
            end
            F.lagged[k, t] = λ
            k += 1
        end
    end
    return nothing
end
