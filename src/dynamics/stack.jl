# Sheets on sheets. In a stack, in the guides and in the box a sheet only ever touches the sheets
# next to it in the stack (the 2D model rests on the same order), so each vertex has two partner
# slots, the sheet below and the sheet above. It finds the point of the partner's surface
# straight across from it by walking a remembered grid cell over its neighbors, and touches it
# as it comes within a few roughnesses of the two half thicknesses, with the same contact law
# (`normal_force`) and the same sticking and sliding friction as on a mesh.
#
# A contact between sheets couples a vertex with the three corners of the partner's triangle,
# and that coupling goes into the step's system whole (`link_rows!`, and on every level of the
# multigrid): treated as against a wall that stays put, both sheets would undo the whole
# overlap, together twice it, and a stack bounces ever harder, even with each side's stiffness
# doubled. Every row is written by its own vertex, gathering from the sheet above, so the sums
# come out the same on every run. Each contact is taken once, from
# the upper sheet's vertices against the lower sheet's triangles (slot 1), and the reaction goes
# to the lower sheet's corners: had both sides each taken half, a contact one side sees and the
# other not yet (the lower sheet resting, the upper landing) would carry half its force, and the
# pair would bounce too. What one side alone misses is an edge of the lower sheet poking up into
# the upper one's face, which a stack does not do.

"""
    stack!(builder, sheets; static, kinetic, stiffness)

Declares `sheets` (pieces of `builder`, bottom first) a stack: each touches the one below and
above it. Office paper on itself: μs = 0.45, μk = 0.38; `stiffness` as for meshes (`add_mesh!`).
"""
function stack!(b::ModelBuilder, sheets::AbstractVector{SheetPiece}; static = 0.45, kinetic = 0.38, stiffness = 2.0e10)
    push!(b.stacks, Stacking([findfirst(==(s), b.sheets) for s in sheets], static, kinetic, stiffness))
    return b
end

"""
The stacks on the device: each sheet's grid, each vertex's sheet, and per vertex two partner
slots (1 below, 2 above) with the partner sheet (0 none), the partner's grid cell nearest to
it, which side of the partner's surface it lies on, and the contact's friction state.
"""
struct Partners{I, I2, F2, V2, U2, F, T2, M2}
    first::I
    nu::I
    nv::I
    sheet::I
    partner::I2          # (2, n)
    cell::I2             # (2, n), 0-based cell of the partner's grid, ci + cj (nu - 1)
    side::F2             # (2, n), +1 or −1 times the partner's normal points toward the vertex
    lagged::F2
    spring::V2
    sliding::U2
    friction::F          # static, kinetic, stiffness per stack, as sheet pairs share them: [3, sheets]
    # Each contact with the sheet below linearized for the step: its normal stiffness and whole
    # block, normal, partner triangle and weights, and the force on the vertex (its partner's
    # corners take it back at their weights).
    link_kn::F2
    link_block::M2
    link_normal::V2
    link_triangle::T2
    link_weights::V2
    link_force::V2
end
Adapt.@adapt_structure Partners

function Partners(backend, b::ModelBuilder)
    up(a) = upload(backend, a)
    n = length(b.x)
    first = Int32[s.first for s in b.sheets]
    sheet = zeros(Int32, n)
    for (k, s) in enumerate(b.sheets), i in vertices(s)
        sheet[i] = k
    end
    partner = zeros(Int32, 2, n)
    cell = zeros(Int32, 2, n)
    friction = zeros(Float32, 3, length(b.sheets))
    for st in b.stacks, (k, s) in enumerate(st.sheets)
        friction[:, s] .= (st.static, st.kinetic, st.stiffness)
        for (slot, other) in ((1, k - 1), (2, k + 1))
            1 <= other <= length(st.sheets) || continue
            p = st.sheets[other]
            g, h = b.sheets[s].grid, b.sheets[p].grid
            for j in 1:g.nv, i in 1:g.nu
                v = vertex(b.sheets[s], i, j)
                partner[slot, v] = p
                # Start at the partner's cell with the same grid position; tracking walks from there.
                ci = clamp(round(Int, (i - 1) * (h.nu - 1) / max(g.nu - 1, 1)), 0, h.nu - 2)
                cj = clamp(round(Int, (j - 1) * (h.nv - 1) / max(g.nv - 1, 1)), 0, h.nv - 2)
                cell[slot, v] = ci + cj * (h.nu - 1)
            end
        end
    end
    z(T, dims...) = KI.zeros(backend, T, dims...)
    return Partners(up(first), up(Int32[s.grid.nu for s in b.sheets]), up(Int32[s.grid.nv for s in b.sheets]), up(sheet), up(partner), up(cell),
        z(Float32, 2, n), z(Float32, 2, n), z(Vec3f, 2, n), z(UInt8, 2, n), up(friction),
        z(Float32, 2, n), z(Mat3f, 2, n), z(Vec3f, 2, n), up(fill((Int32(0), Int32(0), Int32(0)), 2, n)), z(Vec3f, 2, n), z(Vec3f, 2, n))
end

"""
Where the projection of point `p` onto the triangle with corners `a`, `b`, `c` (all relative
to `p`) falls: its barycentric coordinates, the signed distance of `p` from the triangle's
plane along its counterclockwise normal, and that normal.
"""
@inline function projection(a, b, c)
    e1, e2 = b - a, c - a
    scale = max(maximum(abs, e1), maximum(abs, e2), 1.0f-30)
    u1, u2 = e1 / scale, e2 / scale
    n = cross(u1, u2)
    l = norm(n)
    l > 1.0f-20 || return Vec3f(-1, 0, 0), 0.0f0, zero(Vec3f)
    area = n
    n = n / l
    d = -dot(a, n)                         # p is the origin
    q = d * n                              # p − its projection, so the projection is −q relative to p
    w = -q - a                             # projection relative to a
    # Cross products avoid the cancellation in the Gram determinant of a
    # nearly collinear printed triangle; scaling avoids underflow in SI units.
    area2 = dot(area, area)
    v = dot(cross(w / scale, u2), area) / area2
    u = dot(cross(u1, w / scale), area) / area2
    return Vec3f(1 - u - v, v, u), d, n
end

"Corners of cell `cell` of sheet `s`'s grid: (i, j), (i+1, j), (i, j+1), (i+1, j+1)."
@inline function cell_corners(P::Partners, s, cell)
    @inbounds begin
        nu = P.nu[s]
        ci = cell % (nu - Int32(1))
        cj = cell ÷ (nu - Int32(1))
        a = P.first[s] + cj * nu + ci
        return a, a + Int32(1), a + nu, a + nu + Int32(1)
    end
end

"""
The partner's surface straight across from `p`, searched over the cells around `cell`: the
cell and triangle (0: corners a b d, 1: a d c), barycentric weights, distance along the normal,
the normal, and whether `p` lies over the partner at all. A triangle whose projection holds `p`
comes first; failing that one that would within a quarter of a cell beyond its edge: sheets
stacked edge on edge would otherwise lose and regain their edges' support at every µm they
shift against each other, and a vertex that lost it sags into its partner before it returns.
"""
@inline function across(P::Partners, x, p, s, cell)
    @inbounds begin
        nu, nv = P.nu[s], P.nv[s]
        ci0 = cell % (nu - Int32(1))
        cj0 = cell ÷ (nu - Int32(1))
        best = Inf32
        rank = Int32(2)                 # 0: projection inside a triangle, 1: within the margin
        bc, bt = cell, Int32(0)
        bw = zero(Vec3f)
        bd = 0.0f0
        bn = Vec3f(0, 0, 1)
        found = false
        dj = Int32(-1)
        while dj <= Int32(1)
            di = Int32(-1)
            while di <= Int32(1)
                ci, cj = ci0 + di, cj0 + dj
                if Int32(0) <= ci < nu - Int32(1) && Int32(0) <= cj < nv - Int32(1)
                    c = ci + cj * (nu - Int32(1))
                    a, b, cc, d = cell_corners(P, s, c)
                    xa, xb, xc, xd = x[a] - p, x[b] - p, x[cc] - p, x[d] - p
                    w, dist, n = projection(xa, xb, xd)
                    k = ifelse(minimum(w) >= -1.0f-3, Int32(0), ifelse(minimum(w) >= -0.25f0, Int32(1), Int32(2)))
                    if k < rank || (k == rank && k < Int32(2) && abs(dist) < best)
                        rank, best, bc, bt, bw, bd, bn, found = k, abs(dist), c, Int32(0), w, dist, n, true
                    end
                    w, dist, n = projection(xa, xd, xc)
                    k = ifelse(minimum(w) >= -1.0f-3, Int32(0), ifelse(minimum(w) >= -0.25f0, Int32(1), Int32(2)))
                    if k < rank || (k == rank && k < Int32(2) && abs(dist) < best)
                        rank, best, bc, bt, bw, bd, bn, found = k, abs(dist), c, Int32(1), w, dist, n, true
                    end
                end
                di += Int32(1)
            end
            dj += Int32(1)
        end
        return bc, bt, bw, bd, bn, found
    end
end

"Corners of triangle `t` (0: a b d, 1: a d c) of cell `cell` of sheet `s`."
@inline function triangle(P::Partners, s, cell, t)
    a, b, c, d = cell_corners(P, s, cell)
    return a, ifelse(t == Int32(0), b, d), ifelse(t == Int32(0), d, c)
end

"""
How far the point at barycentric weights `w` of triangle (i, j, k) moved from `anchor` to `x`:
from each corner's own move, which is small, so that nothing a cell long is rounded.
"""
@inline moved_by(x, anchor, i, j, k, w) = @inbounds w[1] * (x[i] - anchor[i]) + w[2] * (x[j] - anchor[j]) + w[3] * (x[k] - anchor[k])

"""
Follows each vertex's partner cell as the sheets move: walks it to the nearest cell around it.
Run a few times after building, once per step after that.
"""
function track_partners!(P::Partners, x)
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds for slot in Int32(1):Int32(2)
        s = P.partner[slot, i]
        if s != 0
            c, _, _, _, _, _ = across(P, x, x[i], s, P.cell[slot, i])
            P.cell[slot, i] = c
        end
    end
    return nothing
end

"Which side of its partner's surface each vertex lies on, from the positions at the start."
function partner_sides!(P::Partners, x)
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds for slot in Int32(1):Int32(2)
        s = P.partner[slot, i]
        if s != 0
            _, _, _, d, _, _ = across(P, x, x[i], s, P.cell[slot, i])
            P.side[slot, i] = ifelse(d >= 0, 1.0f0, -1.0f0)
        end
    end
    return nothing
end

"""
Gap between vertex `i` at `p` and its partner in `slot` (both half thicknesses `r` taken off),
the normal pointing from the partner toward the vertex, the partner's triangle and weights, and
whether the vertex lies over the partner at all. The triangle is found in Float32; the gap is
taken again in double-float, since a weight presses a sheet into the next by 0.04 nm: to the
foot point on the triangle, not to a corner, as the normal is only good to Float32 and tilting
it by 10⁻⁷ moves a corner a cell away by 0.2 nm across it. The corners' offsets are taken as
pairs too: on a slope Float32 rounds a cell's edge by 0.1 nm, across the sheet as much as along.
"""
@inline function partner_gap(P::Partners, x, p, i, slot, r, one)
    @inbounds begin
        s = P.partner[slot, i]
        c, t, w, _, n, found = across(P, x, p, s, P.cell[slot, i])
        side = P.side[slot, i]
        a, b, cc = triangle(P, s, c, t)
        dh, dl = dot_pair(n, difference(p, x[a], one), one)
        bh, bl = dot_pair(n, difference(x[b], x[a], one), one)
        ch, cl = dot_pair(n, difference(x[cc], x[a], one), one)
        dl -= w[2] * (bh + bl) + w[3] * (ch + cl) + unit_excess(n, one) / 2 * dh
        g = (side * dh - 2r) + side * dl
        # Deeper than half a sheet: it is past the partner, not pressing into it.
        return g, side * n, a, b, cc, w, found && g > -r
    end
end

"""
Every vertex's contact with the sheet below it, linearized where the sheets stand now: the force
on the vertex (its partner's corners take it back at their weights, `link_rows!`) and the
stiffness, normal, triangle and weights the step's matrix needs. Slip is taken against the
partner's surface at the same weights, which moves too.
"""
function link_contacts!(P::Partners, x, anchor, fixed, area, one, r::Float32, σ::Float32, tangential::Float32)
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds begin
        slot = Int32(1)
        s = P.partner[slot, i]
        kh, B = 0.0f0, zero(Mat3f)
        P.link_force[slot, i] = zero(Vec3f)
        if s != 0 && fixed[i] == 0x00
            p = x[i]
            g, n, a, b, c, w, found = partner_gap(P, x, p, i, slot, r, one[1])
            if found && g < RANGE * σ
                λ, kh = normal_force(g, area[i] * P.friction[3, P.sheet[i]], σ)
                Pt = I3 - n * n'
                u = Pt * ((p - anchor[i]) - moved_by(x, anchor, a, b, c, w))
                ft, Ht = friction_force(Pt, P.spring[slot, i], P.sliding[slot, i], u, P.lagged[slot, i], tangential * kh, P.friction[1, P.sheet[i]], P.friction[2, P.sheet[i]])
                B = kh * (n * n') + Ht
                P.link_force[slot, i] = λ * n + ft
                P.link_normal[slot, i] = n
                P.link_triangle[slot, i] = (a, b, c)
                P.link_weights[slot, i] = w
            end
        end
        P.link_kn[slot, i] = kh
        P.link_block[slot, i] = B
    end
    return nothing
end

"""
The fine level's windows into the neighboring sheets: for each vertex the first corner of its
partner's cell below and above (0 without a partner there).
"""
function anchors_fine!(below, above, P::Partners)
    i = KI.get_global_id(Int32).x
    i <= length(below) || return nothing
    @inbounds begin
        sb, sa = P.partner[1, i], P.partner[2, i]
        below[i] = ifelse(sb == Int32(0), Int32(0), first(cell_corners(P, max(sb, Int32(1)), P.cell[1, i])))
        above[i] = ifelse(sa == Int32(0), Int32(0), first(cell_corners(P, max(sa, Int32(1)), P.cell[2, i])))
    end
    return nothing
end

"The block of a contact with the sheet below: normal and tangential stiffness."
@inline link_block(P::Partners, i) = @inbounds P.link_block[1, i]

@inline link_active(P::Partners, i) = @inbounds P.link_kn[1, i] > 0

"Weight of vertex `x` in the triangle of `u`'s contact below, 0 if it is no corner of it."
@inline function link_weight(P::Partners, u, x)
    @inbounds begin
        a, b, c = P.link_triangle[1, u]
        w = P.link_weights[1, u]
        return ifelse(a == x, w[1], 0.0f0) + ifelse(b == x, w[2], 0.0f0) + ifelse(c == x, w[3], 0.0f0)
    end
end

"""
The contacts between sheets added to the fine level, row by row, each row by its own vertex: as
the upper side its block on the diagonal and −wₖ B into the sheet below (`C`), as a corner of
the triangles of the vertices above it wₖ² B on the diagonal and wₖ wₗ B to the other corners,
and the forces (the vertex's own, and −wₖ f back from each contact it carries). A block to a
held vertex goes into the row sum, as `coupling!` does. `lost` counts couplings the windows do
not hold (sheets that do not lie on one another).
"""
function link_rows!(A, S, C, contact, below, above, nb, held, P::Partners, lost)
    x = KI.get_global_id(Int32).x
    x <= length(below) || return nothing
    @inbounds begin
        for o in Int32(1):Int32(STENCIL)
            C[o, x] = zero(Mat3f)
        end
        if held[x] == 0x00
            f = contact[x]
            if link_active(P, x)
                B = link_block(P, x)
                f += P.link_force[1, x]
                A[CENTER, x] += B
                a, b, c = P.link_triangle[1, x]
                w = P.link_weights[1, x]
                bx = below[x]
                # Written out per corner, as in `assemble_fine!`.
                link_corner!(C, S, nb, held, above, x, bx, a, -w[1] * B, lost)
                link_corner!(C, S, nb, held, above, x, bx, b, -w[2] * B, lost)
                link_corner!(C, S, nb, held, above, x, bx, c, -w[3] * B, lost)
            end
            ax = above[x]
            if ax != 0
                for o in Int32(1):Int32(STENCIL)
                    u = nb[o, ax]
                    if u != 0 && link_active(P, u)
                        wx = link_weight(P, u, x)
                        if wx != 0
                            B = link_block(P, u)
                            f -= wx * P.link_force[1, u]
                            A[CENTER, x] += wx^2 * B
                            a, b, c = P.link_triangle[1, u]
                            w = P.link_weights[1, u]
                            S[1, x] += corner_coupling!(A, nb, held, x, a, wx * w[1] * B)
                            S[1, x] += corner_coupling!(A, nb, held, x, b, wx * w[2] * B)
                            S[1, x] += corner_coupling!(A, nb, held, x, c, wx * w[3] * B)
                        end
                    end
                end
            end
            contact[x] = f
        end
    end
    return nothing
end

"""
The block K between upper vertex `x` and corner `y` of its triangle below: into `C` around `bx`
if `y` is free and its window above holds `x` (so the transposed products find it), else into
the row sum (`y` held) or counted as lost.
"""
@inline function link_corner!(C, S, nb, held, above, x, bx, y, K, lost)
    @inbounds if held[y] != 0x00
        S[1, x] -= K
    else
        o = bx == Int32(0) ? Int32(0) : find_slot(nb, bx, y)
        ay = above[y]
        if o != 0 && ay != 0 && find_slot(nb, ay, x) != 0
            C[o, x] += K
        else
            Atomix.@atomic :monotonic lost[1] += Int32(1)
        end
    end
end

"`coupling!` for a block K between corners `x` and `y` of one triangle: nothing on the diagonal."
@inline corner_coupling!(A, nb, held, x, y, K) = y == x ? zero(Mat3f) : coupling!(A, nb, held, x, y, K)

"Each partner contact at the end of a step: normal force, sticking or sliding (`friction_state`)."
function sheet_friction_update!(P::Partners, x, anchor, area, one, r::Float32, σ::Float32, tangential::Float32)
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds for slot in Int32(1):Int32(1)
        s = P.partner[slot, i]
        λ = 0.0f0
        if s != 0
            p = x[i]
            g, n, a, b, c, w, found = partner_gap(P, x, p, i, slot, r, one[1])
            if found && g < RANGE * σ
                λ, κ = normal_force(g, area[i] * P.friction[3, P.sheet[i]], σ)
                Pt = I3 - n * n'
                u = Pt * ((p - anchor[i]) - moved_by(x, anchor, a, b, c, w))
                P.spring[slot, i], P.sliding[slot, i] = friction_state(Pt * P.spring[slot, i], P.sliding[slot, i], u, λ, tangential * κ, P.friction[1, P.sheet[i]], P.friction[2, P.sheet[i]])
            else
                P.spring[slot, i] = zero(Vec3f)
                P.sliding[slot, i] = 0x00
            end
        end
        P.lagged[slot, i] = λ
    end
    return nothing
end
