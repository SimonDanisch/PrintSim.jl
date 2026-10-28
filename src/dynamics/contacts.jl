# Sheets against meshes. A vertex touches a mesh as it comes within σ of half the sheet's
# thickness, where IPC's log barrier (Li et al. 2020) starts to push the surfaces apart; it grows
# without bound as they close, so contact never needs them to interpenetrate (`normal_force`). Friction is
# a tangential spring capped by Coulomb's law (Cundall and Strack 1979, as the 2D model has it):
# while a vertex sticks, the spring anchored where it stuck holds it, and the step solves it
# exactly; when the spring would pull harder than μs times the normal force it slides, kinetic
# friction μk pulling along the spring, which follows the slip (`friction_force`); when the slip
# turns back, the spring shortens and the contact sticks again. Kinetic friction as a smooth
# force of the slip,
# μk λ u / max(|u|, ε), would hold a sheet starting to slide: every step starts from where the
# velocity alone takes it, where the slip is near zero and the friction a huge viscosity. IPC's
# smooth friction (Li et al. 2020), as Newton's VBD kernels use it, needs Newton iterated to
# convergence: from one linearization it acts as a viscosity, and a sheet on a 20° slope with
# μs = 0.5 creeps at centimeters per second. Contacts with fixed meshes only add to a vertex's
# diagonal block, which the multigrid hierarchy carries down unchanged. Distances are taken from
# the double-float positions (`triangle_gap`): Float32 rounds a position 50 mm out to 4 nm, which
# at σ = 1 µm is 0.4 % of a contact's load.

"Contacts with meshes a vertex keeps at once: a concave corner is two, a fold in a guide more."
const SLOTS = 4

"""
How many activation distances σ apart contacts are tracked. They carry force only below σ; the
margin keeps a contact that one Newton correction closes in the next linearization.
"""
const RANGE = 30

"Static meshes contribute no surface motion; affine coupling specializes these hooks."
struct StaticMeshes end
@inline mesh_motion(::StaticMeshes, C, triangle, p, slot, i, one) = zero(Vec3f)
@inline face_motion(::StaticMeshes, C, kind, feature, edge, xa, xb, xc, slot, t, one) = zero(Vec3f)
@inline cache_mesh_contact!(::StaticMeshes, slot, i, force, block) = nothing

"""
    normal_force(g, k, σ) -> (λ, κ)

IPC's log barrier (Li et al. 2020) on the gap `g`: b(g) = −k (g − σ)² log(g/σ) for 0 < g < σ,
zero from the activation distance σ on, infinite at contact. Returns the force λ = −b′ pushing
the surfaces apart and its stiffness κ = b″, both C¹ at g = σ and unbounded as g → 0. Any load
is therefore carried at a positive gap, and continuous collision detection, which keeps every gap
positive, never works against the contact force: a penalty that needs the surfaces pressed into
each other to carry load (the earlier k σ log(1 + e^(−g/σ)), at most k σ log 2 at touching) left a
loaded idler spring with no collision-free step. `k` (N/m) scales the barrier: the force is
1.19 k σ at g = σ/2 and approaches k σ²/g at small gaps, so k = 2·10⁴ N/m and σ = 1 µm carry a
1 N point load 20 nm apart. Gaps at or below zero, which only rounding of a CCD-safe state can
produce, are evaluated at 10⁻⁶ σ.
"""
@inline function normal_force(g, k, σ)
    g >= σ && return zero(k), zero(k)
    d = max(g, σ * oftype(σ, 1.0e-6))
    r = d - σ
    l = log(d / σ)
    λ = k * (2r * l + r^2 / d)
    κ = -k * (2l + 4r / d - (r / d)^2)
    return λ, κ
end

"Barrier energy b(g), derivative −λ (`normal_force`); infinite at or below contact."
@inline function normal_potential(g, k, σ)
    g >= σ && return zero(k)
    g <= 0 && return oftype(k, Inf)
    return -k * (g - σ)^2 * log(g / σ)
end

"""
    resting_gap(pressure; stiffness = 2e10, roughness = 1e-6) -> m

The gap at which a contact carries `pressure` (Pa) under `normal_force` with `stiffness` (Pa/m)
and activation distance `roughness`: where to lay a sheet so it starts at rest, 0.84 σ above a
plane for 80 g/m² paper under its own weight.
"""
function resting_gap(pressure; stiffness = 2.0e10, roughness = 1.0e-6)
    pressure > 0 || throw(ArgumentError("positive contact pressure required"))
    force(d) = first(normal_force(d, stiffness, roughness))
    lo, hi = roughness * 1e-12, roughness
    for _ in 1:200
        mid = sqrt(lo * hi)
        force(mid) > pressure ? (lo = mid) : (hi = mid)
    end
    return sqrt(lo * hi)
end

"""
Friction force on a contact point slipping by `u` in the tangent plane (projection `Pt`) since
the step started, and its stiffness, from the spring stretched to t = ξ + u: kt t while that
stays within μs (sticking) or μk (sliding) times the last normal force `λ`, else μk λ along t,
with the stiffness μk λ / |t| across it that turns the force with the slip. Sliding fast, t
points along the slip; slowly, along the spring, the way the load pulled; turning back, t
shortens and the contact sticks. Taken at the solve's current positions, not from the last
step: kinetic friction against the last step's slip overshot where the slip was slow and turned
it around, and itself with it, and stacked sheets on 20° (μk = 0.38, tan θ = 0.36) that came
loose slid on faster and faster, their friction swinging across the slope from step to step.
"""
@inline function friction_force(Pt, ξ, sliding, u, λ, kt, μs, μk)
    t = Pt * ξ + u
    l = norm(t)
    kt * l <= ifelse(sliding != 0x00, μk, μs) * λ && return -kt * t, kt * Pt
    d = t / l
    return -μk * λ * d, (μk * λ / l) * (Pt - d * d')
end

"""
The friction spring and whether the contact slides, at the end of a step with slip `u`, normal
force `λ` and tangential stiffness `kt` (`friction_force`): the spring takes up the slip, and
where it would pull harder than μs (sticking) or μk (sliding) times λ, the contact slides and the
spring is cut back to μk λ / kt.
"""
@inline function friction_state(ξ, sliding, u, λ, kt, μs, μk)
    t = ξ + u
    l = norm(t)
    kt * l <= ifelse(sliding != 0x00, μk, μs) * λ && return t, 0x00
    return t * (μk * λ / (kt * l)), 0x01
end

"`v` scaled to unit length; zero for a zero vector."
@inline direction(v) = v / max(norm(v), 1.0f-30)

"""
Each vertex's contacts with meshes, up to `SLOTS`: the mesh (0: none) and triangle, the normal
where the step starts, and what friction carries from step to step: the normal force, the
tangential spring and whether it slides. A contact keeps that state while the vertex stays on
the same mesh with a normal turned by less than 25°.
"""
struct Touches{I, F, V, U}
    mesh::I
    triangle::I
    normal::V
    lagged::F
    spring::V
    sliding::U
end
Adapt.@adapt_structure Touches

function Touches(backend, n)
    z(T) = KI.zeros(backend, T, SLOTS, max(n, 1))
    return Touches(z(Int32), z(Int32), z(Vec3f), z(Float32), z(Vec3f), z(UInt8))
end

"Gap from vertex position `p` to triangle `t` with the sheet's half thickness `r` taken off, and the normal."
@inline function touch_gap(C::Colliders, t, p::Position, r, one)
    g, n, _ = body_gap(C, t, p, one, r)
    return g, n
end

"""
Inserts candidate (`m`, `t`) at rough distance `d` into the `SLOTS` closest so far, kept in
order of distance.
"""
@inline function keep_closest!(ms, ts, ds, m, t, d)
    @inbounds if d < ds[SLOTS]
        k = SLOTS
        while k > 1 && ds[k - 1] > d
            ms[k], ts[k], ds[k] = ms[k - 1], ts[k - 1], ds[k - 1]
            k -= 1
        end
        ms[k], ts[k], ds[k] = m, t, d
    end
    return nothing
end

"""
Finds each vertex's contacts with the meshes where the step starts, within the distance the
step can carry it. The friction state follows each contact that stays on its mesh. A vertex
the inertial prediction carries into a mesh is not set onto it: one held off it by the sheet's
other contacts (a roller's top edge under a sheet's middle row, the sheet resting on the edges
crossing its cells) was set down every step, held there through the step, and pulled the sheet
down with ten times its weight. A vertex landing goes in during the first Newton step and is
taken in by the second (`mesh_forces!`).
"""
function find_touches!(x, inertia, anchor, C::Colliders, T::Touches, fixed, one, r::Float32, σ::Float32)
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds begin
        o = one[1]
        p = x[i]
        ms = MVector{SLOTS, Int32}(ntuple(_ -> Int32(0), SLOTS))
        ts = MVector{SLOTS, Int32}(ntuple(_ -> Int32(0), SLOTS))
        ds = MVector{SLOTS, Float32}(ntuple(_ -> Inf32, SLOTS))
        if fixed[i] == 0x00
            radius = r + RANGE * σ + 2 * (norm(inertia[i] - p) + norm(p - anchor[i])) + 1.0f-4
            q = approximate(p)
            m = Int32(1)
            while m <= Int32(C.meshes)
                root = C.root[m]
                if box_distance2(q, C.lo[root], C.hi[root]) <= radius^2
                    t1, d1, t2, d2 = near_triangles(C, m, p, radius)
                    t1 != 0 && keep_closest!(ms, ts, ds, m, t1, d1)
                    if t2 != 0
                        g1, n1 = touch_gap(C, t1, p, r, o)
                        g2, n2 = touch_gap(C, t2, p, r, o)
                        valid = !(g2 + r < 0 && (g1 + r >= 0 || d2 > d1 + σ))
                        valid && dot(n1, n2) < 0.9999f0 && keep_closest!(ms, ts, ds, m, t2, d2)
                    end
                end
                m += Int32(1)
            end
        end
        # The new contacts, and the friction state of the old ones they continue.
        old_mesh = MVector{SLOTS, Int32}(ntuple(k -> T.mesh[k, i], SLOTS))
        old_normal = MVector{SLOTS, Vec3f}(ntuple(k -> T.normal[k, i], SLOTS))
        old_lagged = MVector{SLOTS, Float32}(ntuple(k -> T.lagged[k, i], SLOTS))
        old_spring = MVector{SLOTS, Vec3f}(ntuple(k -> T.spring[k, i], SLOTS))
        old_sliding = MVector{SLOTS, UInt8}(ntuple(k -> T.sliding[k, i], SLOTS))
        k = 1
        while k <= SLOTS
            t = ts[k]
            n = t == 0 ? zero(Vec3f) : touch_gap(C, t, p, r, o)[2]
            λ, ξ, slides = 0.0f0, zero(Vec3f), 0x00
            j = 1
            while j <= SLOTS
                if t != 0 && old_mesh[j] == ms[k] && dot(old_normal[j], n) > 0.9063f0
                    λ, ξ, slides = old_lagged[j], old_spring[j], old_sliding[j]
                end
                j += 1
            end
            T.mesh[k, i] = ms[k]
            T.triangle[k, i] = t
            T.normal[k, i] = n
            T.lagged[k, i] = λ
            T.spring[k, i] = ξ
            T.sliding[k, i] = slides
            k += 1
        end
    end
    return nothing
end

"""
Contact and friction force of every vertex's mesh contacts, and their stiffness block on its
diagonal, linearized where the vertex stands now. Meshes stand still: the slip is the vertex's
own motion in the tangent plane.
"""
function mesh_forces!(force, block, x, anchor, C::Colliders, T::Touches, fixed, area, one, r::Float32, σ::Float32, tangential::Float32, motion = StaticMeshes())
    i = KI.get_global_id(Int32).x
    i <= length(force) || return nothing
    @inbounds begin
        f = zero(Vec3f)
        H = zero(Mat3f)
        if fixed[i] == 0x00
            k = 1
            while k <= SLOTS
                cache_mesh_contact!(motion, k, i, zero(Vec3f), zero(Mat3f))
                m = T.mesh[k, i]
                if m != 0
                    g, n = touch_gap(C, T.triangle[k, i], x[i], r, one[1])
                    if g < RANGE * σ
                        λ, κ = normal_force(g, area[i] * C.grip[3, m], σ)
                        Pt = I3 - n * n'
                        u = Pt * ((x[i] - anchor[i]) - mesh_motion(motion, C, T.triangle[k, i], x[i], k, i, one[1]))
                        ft, Ht = friction_force(Pt, T.spring[k, i], T.sliding[k, i], u, T.lagged[k, i], tangential * κ, C.grip[1, m], C.grip[2, m])
                        cache_mesh_contact!(motion, k, i, λ * n + ft, κ * (n * n') + Ht)
                        f += λ * n + ft
                        H += κ * (n * n') + Ht
                    end
                end
                k += 1
            end
        end
        force[i] = f
        block[i] = H
    end
    return nothing
end

"""
Triangle `tk` instead of `t` (gap `g`, normal `n`) where it faces the same way as the contact
did (`normal`) and lies closer to `p`.
"""
@inline function closer_face(C::Colliders, tk, t, g, n, normal, p, r, one)
    (tk == 0 || tk == t) && return t, g, n
    gk, nk = touch_gap(C, tk, p, r, one)
    return dot(nk, normal) > 0.9063f0 && abs(gk) < abs(g) ? (tk, gk, nk) : (t, g, n)
end

"""
Every mesh contact at the end of a step: its normal force, and whether it sticks or slides
(`friction_state`). The triangle is found again where the vertex ended: one that slid over an
edge into the next triangle of a flat face is a micrometer from the old one's edge, and its
normal force was lost (a sheet sliding at 80 mm/s lost 0.8 % of its weight that way).
"""
function mesh_friction_update!(x, anchor, C::Colliders, T::Touches, area, one, r::Float32, σ::Float32, tangential::Float32, motion = StaticMeshes())
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds begin
        p = x[i]
        k = 1
        while k <= SLOTS
            m = T.mesh[k, i]
            λ = 0.0f0
            if m != 0
                t = T.triangle[k, i]
                t1, _, t2, _ = near_triangles(C, m, p, r + RANGE * σ + norm(p - anchor[i]) + 1.0f-4)
                g, n = touch_gap(C, t, p, r, one[1])
                t, g, n = closer_face(C, t1, t, g, n, T.normal[k, i], p, r, one[1])
                t, g, n = closer_face(C, t2, t, g, n, T.normal[k, i], p, r, one[1])
                T.triangle[k, i] = t
                T.normal[k, i] = n
                if g < RANGE * σ
                    λ, κ = normal_force(g, area[i] * C.grip[3, m], σ)
                    Pt = I3 - n * n'
                    u = Pt * ((p - anchor[i]) - mesh_motion(motion, C, t, p, k, i, one[1]))
                    T.spring[k, i], T.sliding[k, i] = friction_state(Pt * T.spring[k, i], T.sliding[k, i], u, λ, tangential * κ, C.grip[1, m], C.grip[2, m])
                else
                    T.spring[k, i] = zero(Vec3f)
                    T.sliding[k, i] = 0x00
                end
            end
            T.lagged[k, i] = λ
            k += 1
        end
    end
    return nothing
end
