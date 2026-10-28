using PrintSim, Test, LinearAlgebra
import PrintSim as PS
using GeometryBasics: Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

function deformable_on_plane(backend; steps = 35)
    b = ModelBuilder()
    axes = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))
    add_mesh!(b, box_mesh(Vec3(0.0, 0, -0.005), axes, (0.1, 0.1, 0.005)); static = 0.5)
    points, tets = tet_box((0.01, 0.01, 0.003), (2, 2, 2))
    gap = resting_gap(1000 * 0.003 * 9.81)
    add_deformable!(b, points, tets; density = 1000, young = 1e6,
        place = p -> p + Vec3(0.0, 0, gap))
    solver = WorldSolver(backend, b; iterations = 96)
    state = WorldState(backend, b)
    PS.prime!(solver, state)
    weight = sum(Array(solver.deformables.model.mass)) * 9.81
    loads = Float64[]
    for _ in 1:steps
        PS.step!(solver, state, 1e-3)
        push!(loads, sum(Array(solver.bodies.contacts.lagged)) / weight)
    end
    return (; loads, solver, state)
end

function paper_on_deformable(backend; steps = 12)
    b = ModelBuilder(; gravity = Vec3f(0))
    points, tets = tet_box((0.04, 0.04, 0.004), (2, 2, 2); origin = Vec3(-0.02, -0.02, -0.002))
    add_deformable!(b, points, tets; density = 1000, young = 1e6, static = 0.5, kinetic = 0.4)
    add_sheet!(b, SheetGrid(0.02, 0.02, 9, 9), SheetMaterial();
        place = p -> Vec3(p[1] - 0.01, p[2] - 0.01, 0.002 + 0.5e-4 + 4e-6))
    b.v .= Ref(Vec3f(0.02, 0, 0))
    solver = WorldSolver(backend, b; iterations = 128)
    state = WorldState(backend, b)
    PS.prime!(solver, state)
    precise(p) = Vec3{Float64}(p.hi) + Vec3{Float64}(p.lo)
    fm = Array(solver.deformables.model.mass)
    momentum() = sum(b.mass[i] * precise(v) for (i, v) in enumerate(Array(state.sheets.v))) +
        sum(fm[i] * precise(v) for (i, v) in enumerate(Array(state.deformables.v)))
    before = momentum()
    for _ in 1:steps
        PS.step!(solver, state, 1e-3)
    end
    return (; before, after = momentum(), solver)
end

@testset "sheet–tetrahedron contacts conserve momentum" begin
    result = paper_on_deformable(BACKEND)
    @show result.before result.after
    @test norm(result.after - result.before) < 1e-4 * norm(result.before)
    @test all(iszero, Array(result.solver.fem_contacts.overflow))
end

@testset "tetrahedral surface contacts carry weight" begin
    result = deformable_on_plane(BACKEND)
    @show extrema(result.loads[end-10:end])
    @test isapprox(result.loads[end], 1; rtol = 0.02)
    @test all(isfinite, result.loads)
    @test all(iszero, Array(result.solver.fem_contacts.overflow))
    @test all(iszero, Array(result.solver.bodies.contacts.overflow))
end
