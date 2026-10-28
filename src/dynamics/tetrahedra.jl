# Tetrahedral finite elements for printed leaves and pawls. St Venant–Kirchhoff elasticity
# at small material strain, invariant under finite rotations, with a positive Gauss–Newton
# tangent. Quadratic ten-node tetrahedra avoid the shear locking of linear tetrahedra in
# thin leaves; four integration points integrate their linear bending strains exactly.
# Surface nodes are volume nodes, so contact forces have no extra interpolation.

function add_deformable!(b::ModelBuilder, points, tetrahedra; young = 1.46e9, poisson = 0.35,
        density = 1270.0, fixed = p -> false, place = identity, kw...)
    young > 0 && density > 0 && -1 < poisson < 0.5 || throw(ArgumentError("invalid elastic material"))
    x = Vec3{Float64}.(points)
    linear_tets = NTuple{4, Int32}[]
    boundary = Dict{NTuple{3, Int32}, Vector{NTuple{3, Int32}}}()
    for tuple in tetrahedra
        length(tuple) == 4 && length(unique(tuple)) == 4 || throw(ArgumentError("tetrahedra need four distinct nodes"))
        all(i -> 1 <= i <= length(x), tuple) || throw(ArgumentError("tetrahedron node outside the mesh"))
        a, c, d, e = Int32.(tuple)
        volume = dot(x[c] - x[a], cross(x[d] - x[a], x[e] - x[a])) / 6
        abs(volume) > 0 || throw(ArgumentError("degenerate tetrahedron"))
        volume < 0 && ((d, e) = (e, d))
        t = (a, c, d, e)
        push!(linear_tets, t)
        for face in ((c, d, e), (a, e, d), (a, c, e), (a, d, c))
            key = Tuple(sort(collect(face)))
            push!(get!(boundary, key, NTuple{3, Int32}[]), face)
        end
    end
    isempty(linear_tets) && throw(ArgumentError("a deformable part needs tetrahedra"))
    all(v -> length(v) <= 2, values(boundary)) || throw(ArgumentError("nonmanifold tetrahedral mesh"))
    midpoints = Dict{Tuple{Int32, Int32}, Int32}()
    tets = NTuple{10, Int32}[]
    edges = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    for t in linear_tets
        mids = map(edges) do (a, c)
            get!(midpoints, minmax(t[a], t[c])) do
                push!(x, (x[t[a]] + x[t[c]]) / 2)
                Int32(length(x))
            end
        end
        push!(tets, (t..., mids...))
    end
    faces = NTuple{3, Int32}[]
    for key in sort!(collect(keys(boundary)))
        length(boundary[key]) == 1 || continue
        a, c, d = only(boundary[key])
        ac, cd, da = midpoints[minmax(a, c)], midpoints[minmax(c, d)], midpoints[minmax(d, a)]
        append!(faces, ((a, ac, da), (ac, c, cd), (da, cd, d), (ac, cd, da)))
    end
    nodes = sort!(unique!(reduce(vcat, [collect(f) for f in faces])))
    indices = Dict(n => Int32(i) for (i, n) in enumerate(nodes))
    initial = Vec3{Float64}.(place.(x))
    surface = GeometryBasics.Mesh(Point3d.(initial[nodes]), [GLTriangleFace((indices[n] for n in f)...) for f in faces])
    mesh = add_mesh!(b, surface; kw...)
    push!(b.deformables, TetPiece(mesh, x, initial, tets, nodes, young, poisson, density, UInt8[fixed(p) for p in x]))
    return length(b.deformables)
end

"A conforming six-tetrahedra split of every cell of a rectangular grid, in meters."
function tet_box(size, dims; origin = Vec3(0.0, 0, 0))
    nx, ny, nz = dims
    min(nx, ny, nz) >= 2 || throw(ArgumentError("a tetrahedral grid needs at least two nodes along each axis"))
    node(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
    points = [Vec3{Float64}(origin + Vec3((i - 1) * size[1] / (nx - 1), (j - 1) * size[2] / (ny - 1), (k - 1) * size[3] / (nz - 1)))
        for k in 1:nz for j in 1:ny for i in 1:nx]
    tets = NTuple{4, Int32}[]
    for k in 1:(nz - 1), j in 1:(ny - 1), i in 1:(nx - 1)
        a, h = node(i, j, k), node(i + 1, j + 1, k + 1)
        for path in ((node(i + 1, j, k), node(i + 1, j + 1, k)),
                (node(i + 1, j + 1, k), node(i, j + 1, k)),
                (node(i, j + 1, k), node(i, j + 1, k + 1)),
                (node(i, j + 1, k + 1), node(i, j, k + 1)),
                (node(i, j, k + 1), node(i + 1, j, k + 1)),
                (node(i + 1, j, k + 1), node(i + 1, j, k)))
            push!(tets, Int32.((a, path[1], path[2], h)))
        end
    end
    return points, tets
end

struct FEMModel{P, T, V, F, U, I, K}
    rest::P
    tets::T
    gradient_hi::V
    gradient_lo::V
    stiffness::K
    mass::F
    fixed::U
    start::I
    incidence::I
    corner::I
    surface_node::I
    nodes::Int
    elements::Int
end
Adapt.@adapt_structure FEMModel

function FEMModel(backend, b::ModelBuilder)
    x, tets, gradients = Vec3{Float64}[], NTuple{10, Int32}[], Vec3{Float64}[]
    mass, fixed, stiffness = Float32[], UInt8[], Float32[]
    # Quadrature expands every T10 cell four times. Reserve the known sizes
    # once instead of repeatedly reallocating gigabyte-scale host buffers.
    nodes = sum(p -> length(p.x), b.deformables; init=0)
    cells = sum(p -> length(p.tets), b.deformables; init=0)
    sizehint!(x,nodes); sizehint!(mass,nodes); sizehint!(fixed,nodes)
    sizehint!(tets,4cells); sizehint!(gradients,40cells); sizehint!(stiffness,28cells)
    point_offsets = cumsum([0; [length(m.vertices) for m in b.meshes]])
    surface_node = zeros(Int32, point_offsets[end])
    for piece in b.deformables
        μ = piece.young / (2 * (1 + piece.poisson))
        λ = piece.young * piece.poisson / ((1 + piece.poisson) * (1 - 2piece.poisson))
        offset = length(x)
        append!(x, piece.x); append!(mass, zeros(Float32, length(piece.x))); append!(fixed, piece.fixed)
        for t in piece.tets
            a, c, d, e = t[1:4]
            D = hcat(piece.x[c] - piece.x[a], piece.x[d] - piece.x[a], piece.x[e] - piece.x[a])
            volume = det(D) / 6
            G = inv(D)
            g1, g2, g3 = Vec3{Float64}(G[1, :]), Vec3{Float64}(G[2, :]), Vec3{Float64}(G[3, :])
            base = (-g1 - g2 - g3, g1, g2, g3)
            quad_stiffness=ntuple(i->Float32((i==7 ? λ : 2μ)*volume/4),7)
            # Positive HRZ mass lumping from the exact diagonal of the T10 mass matrix:
            # corner 1/36, edge midpoint 4/27 of the tetrahedron's mass.
            for (corner, node) in enumerate(t)
                mass[offset + node] += piece.density * volume * (corner <= 4 ? 1 / 36 : 4 / 27)
            end
            for point in 1:4
                bary = ntuple(i -> i == point ? 0.5854101966249685 : 0.1381966011250105, 4)
                for i in 1:4
                    push!(gradients, (4bary[i] - 1) * base[i])
                end
                for (i, j) in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
                    push!(gradients, 4 * (bary[i] * base[j] + bary[j] * base[i]))
                end
                push!(tets, t .+ Int32(offset))
                append!(stiffness,quad_stiffness)
            end
        end
        # A mapped proxy may have fewer OR more vertices than the P2 boundary.
        # This one-node table is meaningful only for the unmapped nodal surface;
        # MappedFEMModel supplies its own checked embedding rows afterward.
        if length(b.meshes[piece.mesh].vertices)==length(piece.surface_nodes)
            for (i, node) in enumerate(piece.surface_nodes)
                surface_node[point_offsets[piece.mesh] + i] = offset + node
            end
        end
    end
    start, list, corner = incidence(length(x), tets)
    up(a) = upload(backend, a)
    hi = Vec3f.(gradients)
    lo = Vec3f[Vec3f(g - Vec3{Float64}(h)) for (g, h) in zip(gradients, hi)]
    matrix(a, rows) = isempty(a) ? up(zeros(eltype(a), rows, 1)) : up(reshape(a, rows, :))
    # NTuple's packed Int32 storage already has the desired column layout.
    # Avoid one temporary heap Vector per quadrature cell (millions here).
    connectivity = isempty(tets) ? zeros(Int32, 10, 1) :
        Matrix(reshape(reinterpret(Int32,tets),10,:))
    return FEMModel(up(Position.(x)), up(connectivity), matrix(hi, 10), matrix(lo, 10),
        matrix(stiffness, 7), up(mass), up(fixed), up(start), up(list), up(corner), up(surface_node), length(x), length(tets))
end

struct FEMState{P}
    x::P
    v::P
end
Adapt.@adapt_structure FEMState
function FEMState(backend, b)
    x = Position[p for piece in b.deformables for p in Position.(piece.initial)]
    return FEMState(upload(backend, x), upload(backend, zeros(Position, length(x))))
end

@inline tet_node(t, corner) = ifelse(corner == 1, t[1], ifelse(corner == 2, t[2], ifelse(corner == 3, t[3], t[4])))

function fem_linearize!(force, gradients, block, stress, state, model, one,
        offset::Int32 = Int32(0), count::Int32 = typemax(Int32))
    localid = KI.get_global_id(Int32).x
    localid <= count || return nothing
    e = localid + offset
    e <= model.elements || return nothing
    @inbounds begin
        firstnode = model.tets[1, e]
        f1, f2, f3 = Position(Vec3f(1, 0, 0)), Position(Vec3f(0, 1, 0)), Position(Vec3f(0, 0, 1))
        for corner in Int32(2):Int32(10)
            node = model.tets[corner, e]
            du = minus(difference(state.x[node], state.x[firstnode], one[1]), difference(model.rest[node], model.rest[firstnode], one[1]), one[1])
            gh, gl = model.gradient_hi[corner, e], model.gradient_lo[corner, e]
            f1 = plus(f1, plus(scaled(du, gh[1], one[1]), scaled(du, gl[1], one[1]), one[1]), one[1])
            f2 = plus(f2, plus(scaled(du, gh[2], one[1]), scaled(du, gl[2], one[1]), one[1]), one[1])
            f3 = plus(f3, plus(scaled(du, gh[3], one[1]), scaled(du, gl[3], one[1]), one[1]), one[1])
        end
        T=eltype(eltype(force))
        a1,a2,a3=T===Float64 ?
            (Vec3d(f1.hi)+Vec3d(f1.lo),Vec3d(f2.hi)+Vec3d(f2.lo),Vec3d(f3.hi)+Vec3d(f3.lo)) :
            (approximate(f1),approximate(f2),approximate(f3))
        store_fem_deformation!(gradients,a1,a2,a3,e)
        c1,c2,c3,c4,c5,c6=fem_strain_coefficients(T,f1,f2,f3,one[1])
        for corner in Int32(1):Int32(10)
            force[corner,e]=zero(eltype(force))
            clear_fem_block!(block,corner,e)
        end
        for constraint in Int32(1):Int32(7)
            c = ifelse(constraint == 1, c1, ifelse(constraint == 2, c2, ifelse(constraint == 3, c3,
                ifelse(constraint == 4, c4, ifelse(constraint == 5, c5, ifelse(constraint == 6, c6, c1 + c2 + c3))))))
            first = zero(eltype(force))
            for corner in Int32(2):Int32(10)
                g = T===Float64 ? Vec3d(model.gradient_hi[corner,e])+Vec3d(model.gradient_lo[corner,e]) : model.gradient_hi[corner,e]
                v = fem_constraint_gradient(g,a1,a2,a3,constraint)
                store_fem_gradient!(gradients,v,constraint,corner,e)
                first -= v
            end
            store_fem_gradient!(gradients,first,constraint,Int32(1),e)
            for corner in Int32(1):Int32(10)
                g = gradients[constraint, corner, e]
                k = model.stiffness[constraint, e]
                if eltype(eltype(force))===Float64
                    corner==1 || (force[corner,e]-=Float64(k)*Float64(c)*Vec3d(g))
                else
                    force[corner, e] -= k * c * g
                end
                accumulate_fem_block!(block,g,k,corner,e)
            end
        end
        if eltype(eltype(force))===Float64
            # The relative-coordinate gradient's transpose must balance in the
            # SAME precision. A separately rounded g1=-sum(g2:10) otherwise
            # destroys symmetry in the nearly rigid, ill-conditioned modes.
            value=zero(Vec3d)
            for corner in Int32(2):Int32(10);value-=force[corner,e];end
            force[1,e]=value
        end
        # Store the strain tensor; material stress follows from the two Lamé coefficients.
        root2=sqrt(T(2))
        stress[e] = eltype(stress)(c1,c4/root2,c5/root2,c4/root2,c2,c6/root2,c5/root2,c6/root2,c3)
    end
    return nothing
end

function fem_predict!(state, anchor, inertia, model, one, gravity::Vec3f, dt::Float32)
    i = KI.get_global_id(Int32).x
    i <= model.nodes || return nothing
    @inbounds begin
        p = state.x[i]
        anchor[i] = p
        start = model.fixed[i] == 0x00 ? moved(p, scaled(state.v[i], dt, one[1]), one[1]) : p
        state.x[i] = start
        inertia[i] = model.fixed[i] == 0x00 ? moved(start, dt^2 * gravity, one[1]) : p
    end
    return nothing
end

function fem_rhs!(r, inverse, state, inertia, external, force, block, model, dt::Float32,
        offset::Int32 = Int32(0), count::Int32 = typemax(Int32))
    localid = KI.get_global_id(Int32).x
    localid <= count || return nothing
    i = localid + offset
    i <= model.nodes || return nothing
    @inbounds begin
        inertia_scale=eltype(eltype(r))===Float64 ? Float64(model.mass[i])/Float64(dt)^2 : model.mass[i]/dt^2
        diagonal = inertia_scale * one(eltype(inverse))
        value = inertia_scale * (inertia[i] - state.x[i]) + external[i]
        for at in model.start[i]:(model.start[i + Int32(1)] - Int32(1))
            e, corner = model.incidence[at], model.corner[at] + Int32(1)
            value += force[corner, e]
            diagonal += block[corner, e]
        end
        r[i] = model.fixed[i] == 0x00 ? value : zero(Vec3f)
        inverse[i] = preconditioner_inverse(diagonal)
    end
    return nothing
end

# Apply each quadrature Hessian once, then gather its ten nodal contributions.
# Recomputing all seventy gradient products at every incident node multiplies
# both arithmetic and gradient traffic by ten on the full printed volume.
function fem_apply_elements_gradient_reference!(force, p, gradients, model, offset::Int32 = Int32(0), count::Int32 = typemax(Int32))
    localid = KI.get_global_id(Int32).x
    localid <= count || return nothing
    e = localid + offset
    e <= model.elements || return nothing
    @inbounds begin
        firstnode = model.tets[1, e]
        for corner in Int32(1):Int32(10)
            force[corner, e] = zero(Vec3f)
        end
        for constraint in Int32(1):Int32(7)
            motion = zero(eltype(eltype(p)))
            for corner in Int32(2):Int32(10)
                motion += dot(gradients[constraint, corner, e], p[model.tets[corner, e]] - p[firstnode])
            end
            weighted = model.stiffness[constraint, e] * motion
            for corner in Int32(1):Int32(10)
                if eltype(eltype(force))===Float64
                    if corner!=1
                        value=weighted*Vec3d(gradients[constraint,corner,e])
                        force[corner,e]+=value
                        force[1,e]-=value
                    end
                else
                    force[corner, e] += weighted * gradients[constraint, corner, e]
                end
            end
        end
    end
    return nothing
end

@inline fem_apply_elements!(force,p,gradients,model,offset::Int32=Int32(0),count::Int32=typemax(Int32))=
    fem_apply_elements_gradient_reference!(force,p,gradients,model,offset,count)

# Contract B'KB through the three deformation-gradient columns, rather than
# reconstructing all seven invariant gradients twice for every node. This is
# the same positive Gauss-Newton operator, including the separate trace term.
function fem_apply_elements!(force,p,g::CompactFEMGradients,model,
        offset::Int32=Int32(0),count::Int32=typemax(Int32))
    localid=KI.get_global_id(Int32).x
    localid<=count || return nothing
    e=localid+offset
    e<=model.elements || return nothing
    @inbounds begin
        V=eltype(p);T=eltype(V)
        firstnode=model.tets[1,e]
        u1,u2,u3=zero(V),zero(V),zero(V)
        for j in Int32(2):Int32(10)
            delta=p[model.tets[j,e]]-p[firstnode]
            rest=g.rest_gradient[j,e]
            u1+=rest[1]*delta;u2+=rest[2]*delta;u3+=rest[3]*delta
        end
        a1,a2,a3=g.deformation[1,e],g.deformation[2,e],g.deformation[3,e]
        root2=sqrt(T(2))
        c1,c2,c3=dot(a1,u1),dot(a2,u2),dot(a3,u3)
        c4=(dot(a1,u2)+dot(a2,u1))/root2
        c5=(dot(a1,u3)+dot(a3,u1))/root2
        c6=(dot(a2,u3)+dot(a3,u2))/root2
        trace=model.stiffness[7,e]*(c1+c2+c3)
        s1=model.stiffness[1,e]*c1+trace
        s2=model.stiffness[2,e]*c2+trace
        s3=model.stiffness[3,e]*c3+trace
        s4=model.stiffness[4,e]*c4/root2
        s5=model.stiffness[5,e]*c5/root2
        s6=model.stiffness[6,e]*c6/root2
        v1=s1*a1+s4*a2+s5*a3
        v2=s4*a1+s2*a2+s6*a3
        v3=s5*a1+s6*a2+s3*a3
        first=zero(V)
        for j in Int32(2):Int32(10)
            rest=g.rest_gradient[j,e]
            value=rest[1]*v1+rest[2]*v2+rest[3]*v3
            force[j,e]=value
            first-=value
        end
        force[1,e]=first
    end
    return nothing
end

function fem_apply_gather!(result, p, force, model, dt::Float32, offset::Int32, count::Int32)
    localid = KI.get_global_id(Int32).x
    localid <= count || return nothing
    i = localid + offset
    i <= model.nodes || return nothing
    @inbounds begin
        inertia_scale=eltype(eltype(p))===Float64 ? Float64(model.mass[i])/Float64(dt)^2 : model.mass[i]/dt^2
        value = inertia_scale * p[i]
        for at in model.start[i]:(model.start[i + Int32(1)] - Int32(1))
            value += force[model.corner[at] + Int32(1), model.incidence[at]]
        end
        result[i] = model.fixed[i] == 0x00 ? value : p[i]
    end
    return nothing
end

function fem_apply!(result, p, gradients, model, dt::Float32)
    i = KI.get_global_id(Int32).x
    i <= model.nodes || return nothing
    @inbounds begin
        inertia_scale=eltype(eltype(p))===Float64 ? Float64(model.mass[i])/Float64(dt)^2 : model.mass[i]/dt^2
        value = inertia_scale * p[i]
        for at in model.start[i]:(model.start[i + Int32(1)] - Int32(1))
            e, corner = model.incidence[at], model.corner[at] + Int32(1)
            firstnode = model.tets[1, e]
            for constraint in Int32(1):Int32(7)
            motion = zero(eltype(eltype(p)))
                for k in Int32(2):Int32(10)
                    motion += dot(gradients[constraint, k, e], p[model.tets[k, e]] - p[firstnode])
                end
            if eltype(eltype(p))===Float64 && corner==1
                first=zero(Vec3d)
                for c in Int32(2):Int32(10);first-=Vec3d(gradients[constraint,c,e]);end
                value+=model.stiffness[constraint,e]*motion*first
            else
                value += model.stiffness[constraint, e] * motion * gradients[constraint, corner, e]
            end
            end
        end
        result[i] = model.fixed[i] == 0x00 ? value : p[i]
    end
    return nothing
end

function fem_precondition!(z, r, inverse, model)
    i = KI.get_global_id(Int32).x
    i <= model.nodes || return nothing
    @inbounds z[i] = model.fixed[i] == 0x00 ? inverse[i] * r[i] : zero(Vec3f)
    return nothing
end

function fem_velocity!(state, anchor, model, one, dt::Float32)
    i = KI.get_global_id(Int32).x
    i <= model.nodes || return nothing
    @inbounds state.v[i] = quotient(difference(state.x[i], anchor[i], one[1]), dt, one[1])
    return nothing
end

struct FEMSolver{B, M, P, E, V, G, D, H, F, C, S}
    backend::B
    model::M
    anchor::P
    inertia::P
    force::E
    gradients::G
    block::D
    strain::H
    external::V
    inverse::H
    r::V
    z::V
    p::V
    q::V
    one::F
    scalars::F
    coarse::C
    schwarz::S
    timestep::Float32
    gravity::Vec3f
    newton::Int
    iterations::Int
end

function FEMSolver(backend, b; newton = 4, iterations = 200, timestep = 1e-3, progress = nothing,
        schwarz = false, coarse_builder = FEMCoarse,surface_maps=nothing,compact_gradients=false,
        krylov_precision=:float32,matrix_free_blocks=false)
    krylov_precision in (:float32,:float64) || throw(ArgumentError("select float32 or float64 elastic Krylov arithmetic"))
    krylov_precision!==:float64 || backend===Array || Mantle.supports_float64(Mantle.todevice(backend)) ||
        throw(ArgumentError("FP64 elastic Krylov arithmetic requires native device support"))
    vector_type=krylov_precision===:float64 ? Vec3d : Vec3f
    matrix_type=krylov_precision===:float64 ? Mat{3,3,Float64,9} : Mat3f
    require_surface_maps(b,surface_maps)
    model = FEMModel(backend, b)
    surface_maps===nothing || (model=MappedFEMModel(model,FEMSurfaceTransfer(backend,b,surface_maps)))
    progress === nothing || progress(:elastic_model_ready)
    coarse = if coarse_builder===FEMCoarse && krylov_precision===:float64
        wide_fem_coarse(backend,FEMCoarse(backend,b,timestep;scalar_type=Float64),model.nodes)
    else
        coarse_builder(backend,b,timestep)
    end
    progress === nothing || progress(:elastic_coarse_ready)
    n, e = max(model.nodes, 1), max(model.elements, 1)
    z(T, dims...) = KI.zeros(backend, T, dims...)
    rest_gradients=krylov_precision===:float64 ? WideFEMRestGradients(model.gradient_hi,model.gradient_lo) : model.gradient_hi
    gradients=compact_gradients ? CompactFEMGradients(z(vector_type,3,e),rest_gradients) : z(vector_type,7,10,e)
    blocks=matrix_free_blocks ? ReconstructedFEMBlocks{matrix_type,typeof(gradients),typeof(model.stiffness)}(gradients,model.stiffness) : z(matrix_type,10,e)
    return FEMSolver(backend, model, z(Position, n), z(Position, n), z(vector_type, 10, e), gradients, blocks,
        z(matrix_type, e), z(vector_type, n), z(matrix_type, n), z(vector_type, n), z(vector_type, n), z(vector_type, n), z(vector_type, n),
        upload(backend, Float32[1]), z(Float32, 2iterations + 2), coarse,
        schwarz ? FEMSchwarz(backend,b;scalar_type=eltype(vector_type)) : nothing, Float32(timestep), b.gravity[], newton, iterations)
end

function fem_rhs!(target, s::FEMSolver, state::FEMState, dt::Float32)
    for offset in 0:65536:(s.model.nodes - 1)
        count = min(65536, s.model.nodes - offset)
        launch!(target, fem_rhs!, (s.r, s.inverse, state, s.inertia, s.external, s.force, s.block,
            s.model, dt, Int32(offset), Int32(count)), count)
    end
    return nothing
end

function fem_linearize!(target, s::FEMSolver, state::FEMState)
    for offset in 0:262144:(s.model.elements - 1)
        count = min(262144, s.model.elements - offset)
        launch!(target, fem_linearize!, (s.force, s.gradients, s.block, s.strain, state, s.model,
            s.one, Int32(offset), Int32(count)), count)
    end
    return nothing
end

function fem_product!(target, s::FEMSolver, dt::Float32)
    # The Newton force has already been consumed by fem_rhs!; its storage is
    # scratch during CG and is regenerated at the next Newton linearization.
    for offset in 0:262144:(s.model.elements - 1)
        count = min(262144, s.model.elements - offset)
        launch!(target, fem_apply_elements!, (s.force, s.p, s.gradients, s.model, Int32(offset), Int32(count)), count)
    end
    for offset in 0:65536:(s.model.nodes - 1)
        count = min(65536, s.model.nodes - offset)
        launch!(target, fem_apply_gather!, (s.q, s.p, s.force, s.model, dt, Int32(offset), Int32(count)), count)
    end
    return nothing
end

function step!(target, s::FEMSolver, state::FEMState, dt)
    dt = Float32(dt)
    dt == s.timestep || throw(ArgumentError("construct FEMSolver with timestep=$dt for this step size"))
    n, e = s.model.nodes, s.model.elements
    launch!(target, fem_predict!, (state, s.anchor, s.inertia, s.model, s.one, s.gravity, dt), n)
    for _ in 1:s.newton
        fem_linearize!(target, s, state)
        fem_rhs!(target, s, state, dt)
        fem_prepare_preconditioner!(target,s)
        launch!(target, zero!, (s.scalars,), length(s.scalars))
        fem_precondition!(target, s)
        launch!(target, dot_into!, (s.scalars, s.r, s.z, Int32(n), Int32(1)), cld(n, 256) * 256)
        launch!(target, cg_direction!, (s.p, s.z, s.scalars, Int32(0)), n)
        for k in 1:s.iterations
            fem_product!(target, s, dt)
            launch!(target, dot_into!, (s.scalars, s.p, s.q, Int32(n), Int32(2k)), cld(n, 256) * 256)
            launch!(target, cg_step!, (state.x, s.r, s.p, s.q, s.scalars, s.one, Int32(k)), n)
            fem_precondition!(target, s)
            launch!(target, dot_into!, (s.scalars, s.r, s.z, Int32(n), Int32(2k + 1)), cld(n, 256) * 256)
            launch!(target, cg_direction!, (s.p, s.z, s.scalars, Int32(k)), n)
        end
    end
    launch!(target, fem_velocity!, (state, s.anchor, s.model, s.one, dt), n)
    return state
end
step!(s::FEMSolver, state::FEMState, dt) = step!(s.backend, s, state, dt)
