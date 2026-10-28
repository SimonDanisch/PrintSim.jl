# Implicit Euler for sheets: each step a Newton step on the incremental potential, its linear
# system solved by conjugate gradients preconditioned with a geometric multigrid V-cycle
# (`multigrid.jl`), entirely on the device.
#
# Why not Vertex Block Descent, as Newton's SolverVBD does: paper is light and stiff. At dt = 1 ms
# a vertex's inertia m/dt² is 0.5 N/m against 10⁶ N/m of membrane and 10³ N/m of bending, so a
# sweep moves each vertex a fraction of what the elastic forces ask and smooth bending travels
# across the sheet over thousands of sweeps: a 50 mm paper cantilever hangs at 25 to 50 mm instead
# of 1.8 mm unless the step is 5 µs. Sweeps over whole grid lines do no better (the hinges hold a
# line to its neighbors), conjugate gradients with block-Jacobi need 2160 iterations, with the
# multigrid V-cycle 13 (65×33 vertices, docs/dynamics.md).

"""
Inertial target y = x + dt v + dt² g. The step starts from x + dt v + dt² a, `a` the last
acceleration of a vertex in contact, along its contacts (`velocity!`; `find_touches!` sets a
vertex the prediction carries into a mesh onto it): from the inertial prediction gravity put a resting
vertex 10 µm deep and the right-hand side held 0.4 N of contact force against µN of physics.
The acceleration matters for contacts on a slope: the step linearizes a contact along a Float32
normal, good to 6·10⁻⁸, and 5.6 µm of slide down a 35° face read as 0.3 pm of motion into it,
1 % of what a sheet's weight presses it in (a sheet sliding freely carried 0.8 % less than its
weight). Starting where the last acceleration takes it, the step only solves for the change.
Held vertices stay. Velocities are pairs like the positions: Float32 would round 60 µm of a
step at 60 mm/s to 4 pm, alike for neighbors moving alike.
"""
function forward!(x, v, acceleration, anchor, inertia, fixed, one, g::Vec3f, dt::Float32)
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds begin
        p = x[i]
        anchor[i] = p
        if fixed[i] == 0x00
            start = moved(p, scaled(v[i], dt, one[1]), one[1])
            inertia[i] = moved(start, dt^2 * g, one[1])
            x[i] = moved(start, dt^2 * acceleration[i], one[1])
        else
            inertia[i] = p
        end
    end
    return nothing
end

"Each spring at the current positions: its force on its first vertex and its stiffness block."
function linearize_springs!(force, block, x, anchor, m, dt::Float32)
    e = KI.get_global_id(Int32).x
    e <= length(force) || return nothing
    @inbounds begin
        f, Hd, _ = spring(m, x, anchor, Int32(e), Int32(0), dt)
        force[e] = f
        block[e] = Hd
    end
    return nothing
end

"Each hinge at the current positions: its moment, its stiffness, and the angle's gradients."
function linearize_hinges!(moment, stiffness, g0, g1, g2, g3, x, anchor, m)
    h = KI.get_global_id(Int32).x
    h <= length(moment) || return nothing
    @inbounds begin
        i0, i1, i2, i3 = m.hinges[h]
        o = x[i0]
        θ, d0, d1, d2, d3 = hinge_gradients(zero(Vec3f), x[i1] - o, x[i2] - o, x[i3] - o)
        k = m.hinge_stiffness[h]
        kd = m.hinge_damping[h]
        M = k * (θ - m.hinge_rest[h])
        if kd > 0
            a = anchor[i0]
            M += kd * k * (θ - hinge_angle(zero(Vec3f), anchor[i1] - a, anchor[i2] - a, anchor[i3] - a))
        end
        moment[h] = M
        stiffness[h] = k * (1 + kd)
        g0[h], g1[h], g2[h], g3[h] = d0, d1, d2, d3
    end
    return nothing
end

"""
Linearized sheets: what `linearize_springs!` and `linearize_hinges!` leave for assembling the
step's system.
"""
struct Linearized{V, M, F}
    spring_force::V
    spring_block::M
    hinge_moment::F
    hinge_stiffness::F
    g0::V
    g1::V
    g2::V
    g3::V
end
Adapt.@adapt_structure Linearized

@inline gradient(l::Linearized, h, o) = @inbounds Float32(o == 0) * l.g0[h] + Float32(o == 1) * l.g1[h] + Float32(o == 2) * l.g2[h] + Float32(o == 3) * l.g3[h]

"""
The force at every vertex (the negative gradient of the incremental potential), contacts
included; zero where held.
"""
function gather_force!(r, x, inertia, contact, m, l::Linearized, dt::Float32)
    i = KI.get_global_id(Int32).x
    i <= length(r) || return nothing
    @inbounds begin
        f = -(m.mass[i] / dt^2) * (x[i] - inertia[i]) + contact[i]
        for e in m.vspring_start[i]:(m.vspring_start[i + Int32(1)] - Int32(1))
            s = m.vspring[e]
            f += ifelse(m.vspring_order[e] == Int32(0), l.spring_force[s], -l.spring_force[s])
        end
        for e in m.vhinge_start[i]:(m.vhinge_start[i + Int32(1)] - Int32(1))
            h = m.vhinge[e]
            f -= l.hinge_moment[h] * gradient(l, h, m.vhinge_order[e])
        end
        r[i] = m.fixed[i] == 0x00 ? f : zero(Vec3f)
    end
    return nothing
end

# Each sum over the workgroup (256 lanes) is written out in its kernel: no barrier may sit in a
# loop or a called function.

# The reduction mask is spatially constant. Keep its value in device storage,
# but do not allocate a second full-size energy array just to repeat it.
# Retain dot(a[i], mask), including its IEEE nonfinite propagation.
struct RepeatedEnergyMask{A}
    data::A
end
Adapt.@adapt_structure RepeatedEnergyMask
@inline Base.getindex(a::RepeatedEnergyMask,::Integer)=@inbounds a.data[1]

"`scalars[slot] += a·b` over all vertices."
function dot_into!(scalars, a, b, n::Int32, slot::Int32, ::Val{MODE}=Val(0)) where {MODE}
    i = KI.get_global_id(Int32).x
    li = KI.get_local_id(Int32).x
    T=eltype(scalars)
    sh = KI.localmemory(T, Val((256,)), Val(1))
    value=zero(T)
    if MODE==2
        # The final pass is one group: fixed lane-strided accumulation of
        # partials, then the same group tree as the first pass.
        @inbounds for at in li:Int32(256):n
            value+=a[at]
        end
    else
        @inbounds value=i<=n ? dot(T.(a[i]),T.(b[i])) : zero(T)
    end
    @inbounds sh[li]=value
    KI.barrier()
    @inbounds li <= 128 && (sh[li] += sh[li + 128])
    KI.barrier()
    @inbounds li <= 64 && (sh[li] += sh[li + 64])
    KI.barrier()
    @inbounds li <= 32 && (sh[li] += sh[li + 32])
    KI.barrier()
    @inbounds li <= 16 && (sh[li] += sh[li + 16])
    KI.barrier()
    @inbounds li <= 8 && (sh[li] += sh[li + 8])
    KI.barrier()
    @inbounds li <= 4 && (sh[li] += sh[li + 4])
    KI.barrier()
    @inbounds li <= 2 && (sh[li] += sh[li + 2])
    KI.barrier()
    @inbounds if li==1
        if MODE==1
            scalars[(i-Int32(1))÷Int32(256)+Int32(1)]=sh[1]+sh[2]
        elseif MODE==2
            scalars[slot]+=sh[1]+sh[2]
        else
            Atomix.@atomic scalars[slot]+=sh[1]+sh[2]
        end
    end
    return nothing
end

"x += α p, r −= α q with α = (r·z)ₖ / (p·q)ₖ."
function cg_step!(x, r, p, q, scalars, one, k::Int32)
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds begin
        pq = scalars[2k]
        α = pq > 0 ? scalars[2k - 1] / pq : 0.0f0
        x[i] = moved(x[i], α * p[i], one[1])
        r[i] -= α * q[i]
    end
    return nothing
end

"p = z + β p with β = (r·z)ₖ₊₁ / (r·z)ₖ; p = z for k = 0."
function cg_direction!(p, z, scalars, k::Int32)
    i = KI.get_global_id(Int32).x
    i <= length(p) || return nothing
    @inbounds begin
        if k > 0
            rz = scalars[2k - 1]
            β = rz > 0 ? scalars[2k + 1] / rz : 0.0f0
            p[i] = z[i] + β * p[i]
        else
            p[i] = z[i]
        end
    end
    return nothing
end

"""
Each vertex's velocity over the step, and the acceleration the next step starts with: for a
vertex in contact, along its contacts only (its first with a mesh and the sheet below), so the
start does not carry it into what it rests on.
"""
function velocity!(v, acceleration, x, anchor, T, P, one, dt::Float32)
    i = KI.get_global_id(Int32).x
    i <= length(v) || return nothing
    @inbounds begin
        w = quotient(difference(x[i], anchor[i], one[1]), dt, one[1])
        a = approximate(difference(w, v[i], one[1])) / dt
        touching = false
        if T.lagged[1, i] > 0
            n = T.normal[1, i]
            a -= dot(a, n) * n
            touching = true
        end
        if P.partner[1, i] != 0 && P.link_kn[1, i] > 0
            n = P.link_normal[1, i]
            a -= dot(a, n) * n
            touching = true
        end
        v[i] = w
        # A free vertex starts where its velocity takes it: its acceleration is gravity, which
        # would carry it into what it is about to land on before the contacts see it.
        acceleration[i] = ifelse(touching, a, zero(Vec3f))
    end
    return nothing
end

"""
    SheetSolver(backend, builder; thickness, roughness, tangential, newton, iterations, smooth)

Advances the builder's sheets by implicit Euler: `newton` Newton steps per time step, each
solved by `iterations` conjugate-gradient iterations preconditioned with a multigrid V-cycle
(`smooth` Jacobi sweeps per level and side, the coarsest level solved exactly). The sheets
touch the builder's meshes and each other, their surfaces as rough as `roughness`
(`normal_force`); friction springs are `tangential` times as stiff as the contact. Every Newton
step measures the contacts again where the last one left the sheets.
"""
struct SheetSolver{B, M, X, A, L, G, E, S, C, T, FT, PA}
    backend::B
    model::M
    n::Int
    anchor::X               # positions at the start of the step
    inertia::X              # where inertia and gravity alone would take them
    lin::L
    mg::G
    p::A
    q::A
    extra::E                # extra diagonal blocks per vertex: contacts with meshes
    contact::A              # contact and friction force per vertex
    acceleration::A         # each vertex's last acceleration along its contacts: where the next step starts
    scalars::S
    colliders::C            # the meshes sheets touch: `meshes.jl`
    touches::T              # each vertex's contacts with them: `contacts.jl`
    faces::FT               # their corners and edges pressing into sheets' faces: `faces.jl`
    triangles::Int
    partners::PA            # sheets touching sheets: `stack.jl`
    radius::Float32         # half the sheet's thickness
    roughness::Float32      # of the surfaces sheets touch (m): `normal_force`
    tangential::Float32     # stiffness of the friction springs relative to the contact's
    newton::Int
    iterations::Int
end

function SheetSolver(backend, b::ModelBuilder; thickness = 1.0e-4, roughness = 1.0e-6, tangential = 0.5,
        newton = 2, iterations = 12, smooth = 2, colliders = nothing)
    model = Model(backend, b)
    n = length(b.x)
    z(T, k) = KI.zeros(backend, T, max(k, 1))
    ns, nh = length(b.springs), length(b.hinges)
    lin = Linearized(z(Vec3f, ns), z(Mat3f, ns), z(Float32, nh), z(Float32, nh), z(Vec3f, nh), z(Vec3f, nh), z(Vec3f, nh), z(Vec3f, nh))
    dims = [(s.grid.nu, s.grid.nv) for s in b.sheets]
    stacks = [st.sheets for st in b.stacks]
    tables, coarsest = level_tables(dims, copy(b.fixed), stacks)
    mg = Multigrid(backend, tables, coarsest, stacks; smooth)
    return SheetSolver(backend, model, n, z(Position, n), z(Position, n), lin, mg, z(Vec3f, n), z(Vec3f, n), z(Mat3f, n), z(Vec3f, n), z(Vec3f, n),
        z(Float32, 2iterations + 2), colliders === nothing ? Colliders(backend, b) : colliders, Touches(backend, n), FaceTouches(backend, length(b.tris)), length(b.tris), Partners(backend, b), Float32(thickness / 2), Float32(roughness),
        Float32(tangential), newton, iterations)
end

"Sets every entry of `a` to zero."
function zero!(a)
    i = KI.get_global_id(Int32).x
    i <= length(a) || return nothing
    @inbounds a[i] = zero(eltype(a))
    return nothing
end

"""
    step!(target, solver, state, dt)

Advances `state` by `dt`: launches every kernel on `target`, a backend (they run now) or a
Mantle graph (they run with it).
"""
function predict_sheet!(target, s::SheetSolver, state::State, dt::Float32)
    n, m = s.n, s.model
    launch!(target, forward!, (state.x, state.v, s.acceleration, s.anchor, s.inertia, m.fixed, m.one, m.gravity, dt), n)
    launch!(target, track_partners!, (s.partners, s.anchor), n)
    launch!(target, find_touches!, (state.x, s.inertia, s.anchor, s.colliders, s.touches, m.fixed, m.one, s.radius, s.roughness), n)
    launch!(target, find_face_touches!, (state.x, s.inertia, s.anchor, m.tris, m.tri_edges, s.colliders, s.faces, m.one, s.radius, s.roughness), s.triangles)
    return nothing
end

function linearize_sheet!(target, s::SheetSolver, state::State, dt::Float32, motion = StaticMeshes())
        n, m, lin = s.n, s.model, s.lin
        fine = s.mg.levels[1]
        launch!(target, linearize_springs!, (lin.spring_force, lin.spring_block, state.x, s.anchor, m, dt), length(m.springs))
        launch!(target, linearize_hinges!, (lin.hinge_moment, lin.hinge_stiffness, lin.g0, lin.g1, lin.g2, lin.g3, state.x, s.anchor, m), length(m.hinges))
        launch!(target, mesh_forces!, (s.contact, s.extra, state.x, s.anchor, s.colliders, s.touches, m.fixed, m.area, m.one, s.radius, s.roughness, s.tangential, motion), n)
        launch!(target, face_forces!, (state.x, s.anchor, m.tris, s.colliders, s.faces, m.area, m.one, s.radius, s.roughness, s.tangential, motion), s.triangles)
        P = s.partners
        launch!(target, link_contacts!, (P, state.x, s.anchor, m.fixed, m.area, m.one, s.radius, s.roughness, s.tangential), n)
        launch!(target, anchors_fine!, (fine.below, fine.above, P), n)
        launch!(target, assemble_fine!, (fine.A, fine.S, fine.neighbors, fine.held, s.extra, m, lin, dt), n; group = 64)
        launch!(target, link_rows!, (fine.A, fine.S, fine.C, s.contact, fine.below, fine.above, fine.neighbors, fine.held, P, s.mg.lost), n; group = 64)
        launch!(target, face_rows!, (fine.A, fine.S, s.contact, fine.neighbors, fine.held, m.tris, m.vtri_start, m.vtri, m.vtri_order, s.faces), n; group = 64)
        launch!(target, upper_slots!, (fine.upper, fine.below, fine.above, fine.neighbors, fine.held), n)
        coarsen!(target, s.mg)
        launch!(target, gather_force!, (fine.r, state.x, s.inertia, s.contact, m, lin, dt), n)
        return nothing
end

function step!(target, s::SheetSolver, state::State, dt)
    dt = Float32(dt)
    n, m = s.n, s.model
    fine = s.mg.levels[1]
    r, z = fine.r, fine.z
    groups = cld(n, 256) * 256
    predict_sheet!(target, s, state, dt)
    for _ in 1:s.newton
        linearize_sheet!(target, s, state, dt)
        launch!(target, zero!, (s.scalars,), length(s.scalars))
        vcycle!(target, s.mg)
        launch!(target, dot_into!, (s.scalars, r, z, Int32(n), Int32(1)), groups)
        launch!(target, cg_direction!, (s.p, z, s.scalars, Int32(0)), n)
        for k in 1:s.iterations
            launch!(target, level_apply!, (s.q, s.p, fine), n)
            launch!(target, dot_into!, (s.scalars, s.p, s.q, Int32(n), Int32(2k)), groups)
            launch!(target, cg_step!, (state.x, r, s.p, s.q, s.scalars, m.one, Int32(k)), n)
            vcycle!(target, s.mg)
            launch!(target, dot_into!, (s.scalars, r, z, Int32(n), Int32(2k + 1)), groups)
            launch!(target, cg_direction!, (s.p, z, s.scalars, Int32(k)), n)
        end
    end
    launch!(target, velocity!, (state.v, s.acceleration, state.x, s.anchor, s.touches, s.partners, m.one, dt), n)
    lag!(target, s, state, s.anchor)
    return state
end

"Advances `state` by `dt`, launching every kernel now."
step!(s::SheetSolver, state::State, dt) = step!(s.backend, s, state, dt)

"""
    Stepper(solver, state, dt, steps)

`steps` time steps of `dt` recorded once into a Mantle graph. Coupled worlds replay
one timestep plan because its CG solve already contains a device loop; other
solvers record the timestep batch as a loop. Kernels execute on the device.
"""
struct Stepper{P}
    plan::P
    steps::Int
    dt::Float64
end

function Stepper(s::SheetSolver, state::State, dt, steps)
    g = Mantle.Graph(Mantle.todevice(s.backend))
    Mantle.repeat!(g, steps) do _
        step!(g, s, state, dt)
    end
    plan = Mantle.Plan(g)
    Mantle.record!(plan)
    return Stepper(plan, steps, Float64(dt))
end

"Runs the recorded steps; returns once they are submitted."
run!(st::Stepper) = (Mantle.run!(st.plan); st)

"Wait only for this stepper's submitted batch before observation or changing its inputs."
waitfor!(st::Stepper) = (Mantle.waitfor!(st.plan); st)

"""
    prime!(solver, state; tracking)

Before the first step: finds each vertex's partner cell (`tracking` walks), which side of its
partner it lies on, and the contacts' normal forces at the current positions, which the first
step's friction needs; afterwards every step leaves them for the next.
"""
function prime!(s::SheetSolver, state::State; tracking = 20)
    for _ in 1:tracking
        launch!(s.backend, track_partners!, (s.partners, state.x), s.n)
    end
    launch!(s.backend, partner_sides!, (s.partners, state.x), s.n)
    launch!(s.backend, find_touches!, (state.x, state.x, state.x, s.colliders, s.touches, s.model.fixed, s.model.one, s.radius, s.roughness), s.n)
    launch!(s.backend, find_face_touches!, (state.x, state.x, state.x, s.model.tris, s.model.tri_edges, s.colliders, s.faces, s.model.one, s.radius, s.roughness), s.triangles)
    lag!(s.backend, s, state, state.x)
    return s
end

"The contacts at the end of a step: normal forces and friction states for the next step."
function lag!(target, s::SheetSolver, state::State, anchor = s.anchor)
    m = s.model
    launch!(target, mesh_friction_update!, (state.x, anchor, s.colliders, s.touches, m.area, m.one, s.radius, s.roughness, s.tangential), s.n)
    launch!(target, face_friction_update!, (state.x, anchor, m.tris, s.colliders, s.faces, m.area, m.one, s.radius, s.roughness, s.tangential), s.triangles)
    launch!(target, sheet_friction_update!, (s.partners, state.x, anchor, m.area, m.one, s.radius, s.roughness, s.tangential), s.n)
end
