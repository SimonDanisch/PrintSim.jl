# Geometric multigrid on the sheets' grids, the preconditioner of the conjugate gradients
# (Tamstorf, Jones, McCormick, "Smoothed aggregation multigrid for cloth simulation", 2015, solve
# the same stiff cloth systems with algebraic multigrid; on a structured grid geometric multigrid
# is the simple case of it).
#
# Products take the difference form, (A x)ᵢ = Sᵢ xᵢ + Σⱼ Aᵢⱼ (xⱼ − xᵢ) with Sᵢ the row sum: the
# springs' and hinges' blocks are 10⁵ to 10⁶ times a vertex's inertia, the physics of a sheet in
# motion lies in their null space (moving it rigidly), and Σⱼ Aᵢⱼ xⱼ in Float32 leaves of it noise
# of 5 % of gravity's pull, pointing anywhere (a falling sheet tipped 35° sped up sideways at
# 0.7 m/s²). The row sums of the elastic part and of the contacts between sheets are zero by
# construction, the rest's are restricted down the hierarchy directly.
#
# Every level stores its matrix as a 5×5 stencil of 3×3 blocks around each vertex: a hinge
# reaches two vertices away, and with bilinear prolongation the Galerkin product PᵀAP of a
# radius-2 stencil has radius 2 again. Coarse levels are Galerkin products, not rediscretized
# sheets: for bending (fourth order) bilinear interpolation leaves kinks, and a coarse sheet of
# its own is half as stiff as what the fine grid sees, which costs 635 iterations instead of 13
# (cantilever, 65×33 vertices, dt = 1 ms).
#
# Sheets in a stack touch the sheet below and above, 10⁵ times as stiff as a vertex's inertia,
# so the hierarchy carries those couplings on every level too: each vertex has a second 5×5
# stencil into the sheet below, around the vertex under it, and gathers the transposed blocks
# from the sheet above. A stack bound together that way bends and slides as one: point-wise
# smoothing leaves those motions to the coarsest level, which only sees the longest of them
# (8 sheets, 20°: 122 iterations, condition 375). Each grid position's column through the stack
# is therefore smoothed as one block-tridiagonal system (15 iterations, condition 4.8), and the
# coarsest level is solved exactly over the whole stack.

const STENCIL = 25
"Slot of offset (di, dj), both in −2:2, in a vertex's stencil."
@inline slot(di, dj) = (dj + 2) * 5 + di + 3
const CENTER = Int32(slot(0, 0))

"""
One level of the hierarchy, host side: for each vertex its stencil neighbors (0 outside the
grid), the columns through the stacks, and between this level and the next coarser one, each
vertex's parents with their bilinear weights, each coarse vertex's children and the child at
its own position.
"""
struct LevelTables
    n::Int
    neighbors::Matrix{Int32}        # (STENCIL, n)
    held::Vector{UInt8}
    column_start::Vector{Int32}     # (columns + 1)
    column_member::Vector{Int32}    # (n): each column's vertices, bottom first
    parents::Matrix{Int32}          # (4, n), into the next level; 0 for none
    parent_weights::Matrix{Float32}
    children::Matrix{Int32}         # (9, n of the next level)
    child_weights::Matrix{Float32}
    coincident::Vector{Int32}       # (n of the next level)
end

"The grid size one level coarser: every other vertex."
coarser(n) = (n - 1) ÷ 2 + 1
"Whether a grid can lose every other vertex along both axes."
coarsens(nu, nv) = isodd(nu) && isodd(nv) && nu >= 5 && nv >= 5

"Stencil neighbors of every vertex of sheets with grids `dims`, numbered sheet after sheet."
function stencil_neighbors(dims)
    n = sum(prod, dims)
    nb = zeros(Int32, STENCIL, n)
    first = 0
    for (nu, nv) in dims
        for j in 1:nv, i in 1:nu, dj in -2:2, di in -2:2
            (1 <= i + di <= nu && 1 <= j + dj <= nv) || continue
            nb[slot(di, dj), first + (j - 1) * nu + i] = first + (j + dj - 1) * nu + i + di
        end
        first += nu * nv
    end
    return nb
end

"""
The columns through the stacks of sheets with grids `dims`: for every stack (sheet indices,
bottom first, all of one grid) one column per grid position, its vertices bottom first; every
vertex of a sheet in no stack a column of its own.
"""
function columns(dims, stacks)
    firsts = cumsum([0; [prod(d) for d in dims]])
    instack = falses(length(dims))
    start, member = Int32[1], Int32[]
    for st in stacks
        all(==(dims[st[1]]), (dims[s] for s in st)) || error("the sheets of a stack need one grid: $(dims[st])")
        instack[st] .= true
        for k in 1:prod(dims[st[1]])
            append!(member, Int32.(firsts[s] .+ k for s in st))
            push!(start, length(member) + 1)
        end
    end
    for s in eachindex(dims), k in 1:prod(dims[s])
        instack[s] && continue
        push!(member, firsts[s] + k)
        push!(start, length(member) + 1)
    end
    return start, member
end

"""
Bilinear prolongation from grids `dims` coarsened once: for each fine vertex up to four coarse
parents and weights, for each coarse vertex its children (the transpose) and the child at its
own position.
"""
function prolongation(dims)
    nf = sum(prod, dims)
    cdims = [(coarser(nu), coarser(nv)) for (nu, nv) in dims]
    nc = sum(prod, cdims)
    parents = zeros(Int32, 4, nf)
    pw = zeros(Float32, 4, nf)
    children = [Tuple{Int32, Float32}[] for _ in 1:nc]
    coincident = zeros(Int32, nc)
    w1(i) = isodd(i) ? (((i + 1) ÷ 2, 1.0f0),) : ((i ÷ 2, 0.5f0), (i ÷ 2 + 1, 0.5f0))
    ff, fc = 0, 0
    for ((nu, nv), (Nu, _)) in zip(dims, cdims)
        for j in 1:nv, i in 1:nu
            f = ff + (j - 1) * nu + i
            k = 0
            for (I, wi) in w1(i), (J, wj) in w1(j)
                k += 1
                c = fc + (J - 1) * Nu + I
                parents[k, f] = c
                pw[k, f] = wi * wj
                push!(children[c], (Int32(f), wi * wj))
                wi * wj == 1 && (coincident[c] = f)
            end
        end
        ff += nu * nv
        fc += prod((coarser(nu), coarser(nv)))
    end
    ch = zeros(Int32, 9, nc)
    cw = zeros(Float32, 9, nc)
    for (c, list) in enumerate(children), (k, (f, w)) in enumerate(list)
        ch[k, c] = f
        cw[k, c] = w
    end
    return parents, pw, ch, cw, coincident, cdims
end

"""
The tables of every level for sheets with grids `dims`, held vertices `held` (fine level) and
`stacks`: coarsened while every sheet still can, a coarse vertex held where all its children
are. Returns them and the coarsest level's grids.
"""
function level_tables(dims, held, stacks)
    levels = LevelTables[]
    while true
        n = sum(prod, dims)
        nb = stencil_neighbors(dims)
        cs, cm = columns(dims, stacks)
        if all(d -> coarsens(d...), dims)
            parents, pw, ch, cw, coincident, cdims = prolongation(dims)
            push!(levels, LevelTables(n, nb, held, cs, cm, parents, pw, ch, cw, coincident))
            held = UInt8[all(k -> ch[k, c] == 0 || held[ch[k, c]] != 0x00, 1:9) ? 0x01 : 0x00 for c in 1:size(ch, 2)]
            dims = cdims
        else
            push!(levels, LevelTables(n, nb, held, cs, cm, zeros(Int32, 4, n), zeros(Float32, 4, n), zeros(Int32, 9, 0), zeros(Float32, 9, 0), Int32[]))
            return levels, dims
        end
    end
end

# --- on the device -----------------------------------------------------------------------------

"""
One level on the device: its blocks within the sheets and into the sheet below, where those
are, the factors of its column smoother, its tables, and the vectors a V-cycle needs.
"""
struct Level{M, I, J, U, F, V}
    A::M                # (STENCIL, n): blocks within the sheet; the center one the full diagonal
    S::M                # (1, n): row sums over the free vertices
    C::M                # (STENCIL, n): blocks into the sheet below, around vertex `below`
    below::I            # (n): the vertex of the sheet below at the center of `C`'s stencil, 0: none
    above::I            # (n): the vertex of the sheet above at the center of the stencil `upper` covers
    upper::J            # (STENCIL, n): for each vertex around `above`, this vertex's slot in its `C`
    E::M                # (1, n): column factor, the block to the column's previous vertex
    G::M                # (1, n): column factor, the inverse pivot
    neighbors::J
    held::U
    column_start::I
    column_member::I
    parents::J
    parent_weights::F
    children::J
    child_weights::F
    coincident::I
    r::V                # right-hand side
    z::V                # solution
    t::V                # scratch: the residual, then the column solve's
    n::Int
    columns::Int
end
Adapt.@adapt_structure Level

function Level(backend, t::LevelTables)
    up(a) = upload(backend, a)
    z(T, dims...) = KI.zeros(backend, T, dims...)
    return Level(z(Mat3f, STENCIL, t.n), z(Mat3f, 1, t.n), z(Mat3f, STENCIL, t.n), z(Int32, t.n), z(Int32, t.n), z(Int32, STENCIL, t.n),
        z(Mat3f, 1, t.n), z(Mat3f, 1, t.n), up(t.neighbors), up(t.held), up(t.column_start), up(t.column_member),
        up(t.parents), up(t.parent_weights), up(t.children), up(t.child_weights), up(t.coincident),
        z(Vec3f, t.n), z(Vec3f, t.n), z(Vec3f, t.n), t.n, length(t.column_start) - 1)
end

"Slot of vertex `j` in the stencil around vertex `i`, 0 if it is not there."
@inline function find_slot(nb, i, j)
    @inbounds for o in Int32(1):Int32(STENCIL)
        nb[o, i] == j && return o
    end
    return Int32(0)
end

"""
A block between vertex `i` and its neighbor `j` ≠ `i`: into `i`'s stencil if `j` is free; if `j`
is held it stays put, and the block goes into the row sum over the free vertices (returned).
"""
@inline function coupling!(A, neighbors, held, i, j, B)
    j == i && return zero(Mat3f)
    @inbounds if held[j] == 0x00
        A[find_slot(neighbors, i, j), i] += B
        return zero(Mat3f)
    end
    return -B
end

"""
The fine level's blocks within the sheets from the linearized sheets: the couplings of springs
and hinges, the row sums (inertia, `extra` diagonal blocks for contacts with shapes, and the
couplings to held vertices), and the full diagonal. Held vertices get an identity row and no
couplings. The contacts between sheets come on top (`link_rows!`).
"""
function assemble_fine!(A, S, neighbors, held, extra, m, l::Linearized, dt::Float32)
    i = KI.get_global_id(Int32).x
    i <= length(held) || return nothing
    @inbounds begin
        for o in Int32(1):Int32(STENCIL)
            A[o, i] = zero(Mat3f)
        end
        if held[i] != 0x00
            A[CENTER, i] = I3
            S[1, i] = I3
        else
            R = (m.mass[i] / dt^2) * I3 + extra[i]
            for e in m.vspring_start[i]:(m.vspring_start[i + Int32(1)] - Int32(1))
                s = m.vspring[e]
                a, b = m.springs[s]
                R += coupling!(A, neighbors, held, i, ifelse(a == i, b, a), -l.spring_block[s])
            end
            for e in m.vhinge_start[i]:(m.vhinge_start[i + Int32(1)] - Int32(1))
                h = m.vhinge[e]
                j0, j1, j2, j3 = m.hinges[h]
                k = l.hinge_stiffness[h]
                g = gradient(l, h, m.vhinge_order[e])
                # Written out per corner: a loop over the tuple indexes it at run time, and RADV's
                # shader compiler cannot store through the private array that becomes.
                R += coupling!(A, neighbors, held, i, j0, k * (g * l.g0[h]'))
                R += coupling!(A, neighbors, held, i, j1, k * (g * l.g1[h]'))
                R += coupling!(A, neighbors, held, i, j2, k * (g * l.g2[h]'))
                R += coupling!(A, neighbors, held, i, j3, k * (g * l.g3[h]'))
            end
            S[1, i] = R
            D = R
            for o in Int32(1):Int32(STENCIL)
                o != CENTER && (D -= A[o, i])
            end
            A[CENTER, i] = D
        end
    end
    return nothing
end

"""
For every vertex of the coarse level its windows into the sheets below and above: around the
parent of what the fine vertex at its position has, so that the coarse blocks of a stack whose
sheets lie on one another stay within two vertices of it.
"""
function coarse_anchors!(belowc, abovec, belowf, abovef, coincident, parents)
    I = KI.get_global_id(Int32).x
    I <= length(belowc) || return nothing
    @inbounds begin
        f = coincident[I]
        b, a = belowf[f], abovef[f]
        belowc[I] = ifelse(b == Int32(0), Int32(0), parents[1, max(b, Int32(1))])
        abovec[I] = ifelse(a == Int32(0), Int32(0), parents[1, max(a, Int32(1))])
    end
    return nothing
end

"""
Coarse level blocks as the Galerkin product PᵀAP of the finer one, one coarse vertex per thread:
its children, their blocks, and the parents of their neighbors, within its sheet and into the
sheet below. P leaves held vertices out, so they take no part. A block into the sheet below is
kept only where the vertex there finds it again from its side (its window above holds this
vertex): the transposed products gather it from there, so the level stays symmetric.
"""
function galerkin!(Ac, Sc, Cc, heldc, nbc, belowc, abovec, Af, Sf, Cf, nbf, belowf, heldf, children, child_weights, parents, parent_weights)
    I = KI.get_global_id(Int32).x
    I <= length(heldc) || return nothing
    @inbounds begin
        for o in Int32(1):Int32(STENCIL)
            Ac[o, I] = zero(Mat3f)
            Cc[o, I] = zero(Mat3f)
        end
        if heldc[I] != 0x00
            Ac[CENTER, I] = I3
            Sc[1, I] = I3
        else
            R = zero(Mat3f)
            bI = belowc[I]
            for k in Int32(1):Int32(9)
                f = children[k, I]
                if f != 0 && heldf[f] == 0x00
                    wf = child_weights[k, I]
                    R += wf * Sf[1, f]
                    for o in Int32(1):Int32(STENCIL)
                        j = nbf[o, f]
                        if j != 0 && heldf[j] == 0x00
                            B = wf * Af[o, f]
                            for q in Int32(1):Int32(4)
                                J = parents[q, j]
                                if J != 0 && heldc[J] == 0x00
                                    Ac[find_slot(nbc, I, J), I] += parent_weights[q, j] * B
                                end
                            end
                        end
                    end
                    bf = belowf[f]
                    if bf != 0 && bI != 0
                        for o in Int32(1):Int32(STENCIL)
                            j = nbf[o, bf]
                            if j != 0 && heldf[j] == 0x00
                                B = wf * Cf[o, f]
                                for q in Int32(1):Int32(4)
                                    J = parents[q, j]
                                    if J != 0 && heldc[J] == 0x00
                                        oc = find_slot(nbc, bI, J)
                                        aJ = abovec[J]
                                        if oc != 0 && aJ != 0 && find_slot(nbc, aJ, I) != 0
                                            Cc[oc, I] += parent_weights[q, j] * B
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            Sc[1, I] = R
        end
    end
    return nothing
end

"""
For each vertex and each vertex of the window above it, this vertex's slot in that one's
blocks below (0 where it has none): what the transposed products gather.
"""
function upper_slots!(upper, below, above, nb, held)
    i = KI.get_global_id(Int32).x
    i <= length(below) || return nothing
    @inbounds begin
        a = above[i]
        for o in Int32(1):Int32(STENCIL)
            u = ifelse(a == Int32(0) || held[i] != 0x00, Int32(0), nb[o, max(a, Int32(1))])
            b = ifelse(u == Int32(0), Int32(0), below[max(u, Int32(1))])
            upper[o, i] = ifelse(b == Int32(0), Int32(0), find_slot(nb, max(b, Int32(1)), i))
        end
    end
    return nothing
end

"(K x)ᵢ for a level's blocks, in the difference form: within the sheet, into the sheet below, and the transposed blocks of the sheet above."
@inline function level_times(A, S, C, upper, nb, below, above, x, i)
    @inbounds begin
        xi = x[i]
        y = S[1, i] * xi
        for o in Int32(1):Int32(STENCIL)
            j = nb[o, i]
            (j != 0 && o != CENTER) && (y += A[o, i] * (x[j] - xi))
        end
        b = below[i]
        if b != 0
            for o in Int32(1):Int32(STENCIL)
                j = nb[o, b]
                j != 0 && (y += C[o, i] * (x[j] - xi))
            end
        end
        a = above[i]
        if a != 0
            for o in Int32(1):Int32(STENCIL)
                k = upper[o, i]
                if k != 0
                    u = nb[o, a]
                    y += C[k, u]' * (x[u] - xi)
                end
            end
        end
        return y
    end
end

@inline level_times(L, x, i) = level_times(L.A, L.S, L.C, L.upper, L.neighbors, L.below, L.above, x, i)

"y = K x on a level."
function level_apply!(y, x, L)
    i = KI.get_global_id(Int32).x
    i <= length(y) || return nothing
    @inbounds y[i] = level_times(L, x, i)
    return nothing
end

"t = r − K z on a level."
function residual!(t, z, r, L)
    i = KI.get_global_id(Int32).x
    i <= length(t) || return nothing
    @inbounds t[i] = r[i] - level_times(L, z, i)
    return nothing
end

"""
Each column's block-tridiagonal system (its vertices' diagonal blocks and the blocks between
neighbors in the stack) factored: the block to the previous vertex and the inverse pivot.
"""
function column_factor!(E, G, A, C, nb, below, column_start, column_member)
    c = KI.get_global_id(Int32).x
    c < length(column_start) || return nothing
    @inbounds begin
        k = column_start[c]
        last = column_start[c + Int32(1)]
        prev = Int32(0)
        Gp = zero(Mat3f)
        while k < last
            v = column_member[k]
            D = A[CENTER, v]
            Ev = zero(Mat3f)
            b = below[v]
            if prev != 0 && b != 0
                o = find_slot(nb, b, prev)
                o != 0 && (Ev = C[o, v])
            end
            Gv = inv(D - Ev * Gp * Ev')
            E[1, v] = Ev
            G[1, v] = Gv
            Gp = Gv
            prev = v
            k += Int32(1)
        end
    end
    return nothing
end

"""
`t` overwritten with T⁻¹ t for every column's block T, and `z += ω t`; `fresh` starts `z` at
`ω t` instead.
"""
function column_sweep!(z, t, E, G, column_start, column_member, ω::Float32, fresh::Bool)
    c = KI.get_global_id(Int32).x
    c < length(column_start) || return nothing
    @inbounds begin
        first = column_start[c]
        last = column_start[c + Int32(1)] - Int32(1)
        # Forward: yₘ = tₘ − Eₘ Gₘ₋₁ yₘ₋₁.
        k = first
        prev = Int32(0)
        while k <= last
            v = column_member[k]
            prev != 0 && (t[v] -= E[1, v] * (G[1, prev] * t[prev]))
            prev = v
            k += Int32(1)
        end
        # Backward: xₘ = Gₘ (yₘ − Eₘ₊₁ᵀ xₘ₊₁).
        k = last
        next = Int32(0)
        while k >= first
            v = column_member[k]
            y = t[v]
            next != 0 && (y -= E[1, next]' * t[next])
            t[v] = G[1, v] * y
            next = v
            k -= Int32(1)
        end
        k = first
        while k <= last
            v = column_member[k]
            z[v] = ifelse(fresh, zero(Vec3f), z[v]) + ω * t[v]
            k += Int32(1)
        end
    end
    return nothing
end

"Coarse right-hand side: rc = Pᵀ t, held vertices left out."
function restrict!(rc, t, children, child_weights, heldc)
    I = KI.get_global_id(Int32).x
    I <= length(rc) || return nothing
    @inbounds begin
        s = zero(Vec3f)
        if heldc[I] == 0x00
            for k in Int32(1):Int32(9)
                f = children[k, I]
                f != 0 && (s += child_weights[k, I] * t[f])
            end
        end
        rc[I] = s
    end
    return nothing
end

"z += P zc, held vertices left out."
function prolong!(z, zc, parents, parent_weights, held)
    i = KI.get_global_id(Int32).x
    i <= length(z) || return nothing
    @inbounds if held[i] == 0x00
        s = zero(Vec3f)
        for q in Int32(1):Int32(4)
            J = parents[q, i]
            J != 0 && (s += parent_weights[q, i] * zc[J])
        end
        z[i] += s
    end
    return nothing
end

"""
The coarsest level solved exactly, stack by stack (a sheet in no stack is a stack of its own):
its sheets bottom first, each sheet's first vertex and vertex count, and where its dense
Cholesky factor, its block toward the sheet below and its right-hand side start in `scratch`.
Jacobi cannot do it: moving a sheet rigidly in its plane costs only inertia against membrane
blocks a thousand times stiffer, and each sweep removes a millionth of that error (a falling
sheet tipped 35° fell in its plane at 4.48 instead of 4.61 m/s² with 12 iterations, exactly
with 60).
"""
struct Coarsest{I, F}
    stack_start::I      # (stacks + 1) into `sheet`
    first::I            # per sheet, bottom first within each stack
    count::I
    factor::I           # where its factor starts in `scratch`
    coupling::I         # where its block toward the sheet below starts
    rhs::I
    scratch::F
    stacks::Int
end
Adapt.@adapt_structure Coarsest

function Coarsest(backend, dims, stacks)
    up(a) = upload(backend, a)
    firsts = cumsum([1; [prod(d) for d in dims]])[1:(end - 1)]
    order = [st for st in stacks]
    instack = falses(length(dims))
    foreach(st -> instack[st] .= true, stacks)
    append!(order, [[s] for s in eachindex(dims) if !instack[s]])
    start, first, count, factor, coupling, rhs = Int32[1], Int32[], Int32[], Int32[], Int32[], Int32[]
    at = 1
    for st in order
        prevn = 0
        for s in st
            N = 3 * prod(dims[s])
            push!(first, firsts[s]); push!(count, prod(dims[s]))
            push!(factor, at); at += N * N
            push!(coupling, at); at += N * prevn
            push!(rhs, at); at += N
            prevn = N
        end
        push!(start, length(first) + 1)
    end
    return Coarsest(up(start), up(first), up(count), up(factor), up(coupling), up(rhs), KI.zeros(backend, Float32, at), length(order))
end

"Writes block `B` at block row `a`, column `b` of the column-major matrix at `base` with `N` rows."
@inline function dense_block!(M, base, N, a, b, B)
    @inbounds begin
        r, c = Int32(3) * a, Int32(3) * b
        M[base + c * N + r] = B[1, 1]; M[base + c * N + r + Int32(1)] = B[2, 1]; M[base + c * N + r + Int32(2)] = B[3, 1]
        M[base + (c + Int32(1)) * N + r] = B[1, 2]; M[base + (c + Int32(1)) * N + r + Int32(1)] = B[2, 2]; M[base + (c + Int32(1)) * N + r + Int32(2)] = B[3, 2]
        M[base + (c + Int32(2)) * N + r] = B[1, 3]; M[base + (c + Int32(2)) * N + r + Int32(1)] = B[2, 3]; M[base + (c + Int32(2)) * N + r + Int32(2)] = B[3, 3]
    end
end

"Lower-triangular solve L y = b in place, `L` at `base` (N×N), `b` at `x` with stride 1."
@inline function lower_solve!(M, base, N, x)
    @inbounds begin
        i = Int32(0)
        while i < N
            v = M[x + i]
            k = Int32(0)
            while k < i
                v -= M[base + k * N + i] * M[x + k]
                k += Int32(1)
            end
            M[x + i] = v / M[base + i * N + i]
            i += Int32(1)
        end
    end
end

"Upper-triangular solve Lᵀ y = b in place, `L` at `base` (N×N)."
@inline function upper_solve!(M, base, N, x)
    @inbounds begin
        i = N - Int32(1)
        while i >= Int32(0)
            v = M[x + i]
            k = i + Int32(1)
            while k < N
                v -= M[base + i * N + k] * M[x + k]
                k += Int32(1)
            end
            M[x + i] = v / M[base + i * N + i]
            i -= Int32(1)
        end
    end
end

"""
Each stack's coarsest system factored, one thread per stack: per sheet s (bottom first) its
dense blocks K_ss and C_s (toward the sheet below), Wₛ = Cₛ Lₛ₋₁⁻ᵀ and Lₛ Lₛᵀ = K_ss − Wₛ Wₛᵀ,
so that the stack's matrix is L Lᵀ with L block-bidiagonal. While loops over Int32 only.
"""
function coarse_factor!(A, C, nb, below, held, cs::Coarsest)
    s = KI.get_global_id(Int32).x
    s <= cs.stacks || return nothing
    @inbounds begin
        M = cs.scratch
        q = cs.stack_start[s]
        prev = Int32(0)
        while q < cs.stack_start[s + Int32(1)]
            f0, nv = cs.first[q], cs.count[q]
            N = Int32(3) * nv
            base = cs.factor[q]
            k = Int32(0)
            while k < N * N
                M[base + k] = 0.0f0
                k += Int32(1)
            end
            a = Int32(0)
            while a < nv
                I = f0 + a
                if held[I] != 0x00
                    dense_block!(M, base, N, a, a, I3)
                else
                    o = Int32(1)
                    while o <= Int32(STENCIL)
                        J = nb[o, I]
                        J != 0 && dense_block!(M, base, N, a, J - f0, A[o, I])
                        o += Int32(1)
                    end
                end
                a += Int32(1)
            end
            if prev != 0
                # Wₛ = Cₛ Lₛ₋₁⁻ᵀ, row by row: each row of Cₛ solved against Lₛ₋₁ (Lₛ₋₁ wᵀ = cᵀ).
                pf, pn = cs.first[prev], cs.count[prev]
                P = Int32(3) * pn
                W = cs.coupling[q]
                k = Int32(0)
                while k < N * P
                    M[W + k] = 0.0f0
                    k += Int32(1)
                end
                a = Int32(0)
                while a < nv
                    I = f0 + a
                    b = below[I]
                    if held[I] == 0x00 && b != 0
                        o = Int32(1)
                        while o <= Int32(STENCIL)
                            J = nb[o, b]
                            J != 0 && dense_block!(M, W, N, a, J - pf, C[o, I])
                            o += Int32(1)
                        end
                    end
                    a += Int32(1)
                end
                # Row r of Wₛ is Lₛ₋₁⁻¹ (row r of Cₛ)ᵀ: copied out (it is strided by N), solved in
                # the sheet below's right-hand side space, copied back.
                x = cs.rhs[prev]
                r = Int32(0)
                while r < N
                    k = Int32(0)
                    while k < P
                        M[x + k] = M[W + k * N + r]
                        k += Int32(1)
                    end
                    lower_solve!(M, cs.factor[prev], P, x)
                    k = Int32(0)
                    while k < P
                        M[W + k * N + r] = M[x + k]
                        k += Int32(1)
                    end
                    r += Int32(1)
                end
                # K_ss − Wₛ Wₛᵀ, lower triangle.
                j = Int32(0)
                while j < N
                    i = j
                    while i < N
                        v = M[base + j * N + i]
                        k = Int32(0)
                        while k < P
                            v -= M[W + k * N + i] * M[W + k * N + j]
                            k += Int32(1)
                        end
                        M[base + j * N + i] = v
                        i += Int32(1)
                    end
                    j += Int32(1)
                end
            end
            # Cholesky, lower triangle in place.
            j = Int32(0)
            while j < N
                d = M[base + j * N + j]
                k = Int32(0)
                while k < j
                    d -= M[base + k * N + j]^2
                    k += Int32(1)
                end
                d = sqrt(max(d, 1.0f-30))
                M[base + j * N + j] = d
                i = j + Int32(1)
                while i < N
                    v = M[base + j * N + i]
                    k = Int32(0)
                    while k < j
                        v -= M[base + k * N + i] * M[base + k * N + j]
                        k += Int32(1)
                    end
                    M[base + j * N + i] = v / d
                    i += Int32(1)
                end
                j += Int32(1)
            end
            prev = q
            q += Int32(1)
        end
    end
    return nothing
end

"""
Each stack's coarsest system solved with its factor: forward through the sheets bottom first
(Lₛ yₛ = bₛ − Wₛ yₛ₋₁), back top first (Lₛᵀ xₛ = yₛ − Wₛ₊₁ᵀ xₛ₊₁).
"""
function coarse_solve!(z, r, cs::Coarsest)
    s = KI.get_global_id(Int32).x
    s <= cs.stacks || return nothing
    @inbounds begin
        M = cs.scratch
        q0, q1 = cs.stack_start[s], cs.stack_start[s + Int32(1)] - Int32(1)
        q = q0
        while q <= q1
            f0, N = cs.first[q], Int32(3) * cs.count[q]
            x = cs.rhs[q]
            a = Int32(0)
            while a < cs.count[q]
                v = r[f0 + a]
                M[x + Int32(3) * a] = v[1]; M[x + Int32(3) * a + Int32(1)] = v[2]; M[x + Int32(3) * a + Int32(2)] = v[3]
                a += Int32(1)
            end
            if q > q0
                W, xp, P = cs.coupling[q], cs.rhs[q - Int32(1)], Int32(3) * cs.count[q - Int32(1)]
                i = Int32(0)
                while i < N
                    v = M[x + i]
                    k = Int32(0)
                    while k < P
                        v -= M[W + k * N + i] * M[xp + k]
                        k += Int32(1)
                    end
                    M[x + i] = v
                    i += Int32(1)
                end
            end
            lower_solve!(M, cs.factor[q], N, x)
            q += Int32(1)
        end
        q = q1
        while q >= q0
            N = Int32(3) * cs.count[q]
            x = cs.rhs[q]
            if q < q1
                W, xn, Nn = cs.coupling[q + Int32(1)], cs.rhs[q + Int32(1)], Int32(3) * cs.count[q + Int32(1)]
                k = Int32(0)
                while k < N
                    v = M[x + k]
                    i = Int32(0)
                    while i < Nn
                        v -= M[W + k * Nn + i] * M[xn + i]
                        i += Int32(1)
                    end
                    M[x + k] = v
                    k += Int32(1)
                end
            end
            upper_solve!(M, cs.factor[q], N, x)
            q -= Int32(1)
        end
        q = q0
        while q <= q1
            f0, x = cs.first[q], cs.rhs[q]
            a = Int32(0)
            while a < cs.count[q]
                z[f0 + a] = Vec3f(M[x + Int32(3) * a], M[x + Int32(3) * a + Int32(1)], M[x + Int32(3) * a + Int32(2)])
                a += Int32(1)
            end
            q += Int32(1)
        end
    end
    return nothing
end

"""
    Multigrid(backend, tables, coarsest, stacks; smooth, ω)

The hierarchy on the device and how a V-cycle runs: `smooth` column sweeps before and after
the coarse correction, damping `ω`, the coarsest level solved exactly. `lost` counts couplings
between sheets the fine level could not place (a stack whose sheets do not lie on one another).
"""
struct Multigrid{L, C, N}
    levels::Vector{L}
    coarsest::C
    lost::N
    smooth::Int
    ω::Float32
end
Multigrid(backend, tables::Vector{LevelTables}, dims, stacks; smooth = 2, ω = 0.6) =
    Multigrid([Level(backend, t) for t in tables], Coarsest(backend, dims, stacks), KI.zeros(backend, Int32, 1), smooth, Float32(ω))

"""
After the fine level was assembled: the windows into the neighboring sheets, the Galerkin
products down the hierarchy, every level's column factors and the coarsest level's factor.
"""
function coarsen!(target, mg::Multigrid)
    for k in 1:(length(mg.levels) - 1)
        f, c = mg.levels[k], mg.levels[k + 1]
        launch!(target, coarse_anchors!, (c.below, c.above, f.below, f.above, f.coincident, f.parents), c.n)
        launch!(target, galerkin!, (c.A, c.S, c.C, c.held, c.neighbors, c.below, c.above, f.A, f.S, f.C, f.neighbors, f.below, f.held, f.children, f.child_weights,
            f.parents, f.parent_weights), c.n; group = 64)
        launch!(target, upper_slots!, (c.upper, c.below, c.above, c.neighbors, c.held), c.n)
    end
    for L in mg.levels
        launch!(target, column_factor!, (L.E, L.G, L.A, L.C, L.neighbors, L.below, L.column_start, L.column_member), L.columns; group = 64)
    end
    L = mg.levels[end]
    launch!(target, coarse_factor!, (L.A, L.C, L.neighbors, L.below, L.held, mg.coarsest), mg.coarsest.stacks; group = 1)
end

"`sweeps` column sweeps on level `L` ending in `L.z`, from zero or (`fresh = false`) from `L.z`."
function smooth!(target, L::Level, sweeps, ω, fresh)
    for k in 1:sweeps
        if k == 1 && fresh
            launch!(target, copy_into!, (L.t, L.r), L.n)
        else
            launch!(target, residual!, (L.t, L.z, L.r, L), L.n)
        end
        launch!(target, column_sweep!, (L.z, L.t, L.E, L.G, L.column_start, L.column_member, ω, k == 1 && fresh), L.columns; group = 64)
    end
end

"a .= b"
function copy_into!(a, b)
    i = KI.get_global_id(Int32).x
    i <= length(a) || return nothing
    @inbounds a[i] = b[i]
    return nothing
end

"""
One V-cycle from level `k`: solves A z ≈ r for the r in `levels[k].r` into `levels[k].z`.
Symmetric (the same sweeps before and after, restriction the transpose of prolongation), so it
serves as a preconditioner for conjugate gradients.
"""
function vcycle!(target, mg::Multigrid, k = 1)
    L = mg.levels[k]
    if k == length(mg.levels)
        launch!(target, coarse_solve!, (L.z, L.r, mg.coarsest), mg.coarsest.stacks; group = 1)
        return
    end
    C = mg.levels[k + 1]
    smooth!(target, L, mg.smooth, mg.ω, true)
    launch!(target, residual!, (L.t, L.z, L.r, L), L.n)
    launch!(target, restrict!, (C.r, L.t, L.children, L.child_weights, C.held), C.n)
    vcycle!(target, mg, k + 1)
    launch!(target, prolong!, (L.z, C.z, L.parents, L.parent_weights, L.held), L.n)
    smooth!(target, L, mg.smooth, mg.ω, false)
end
