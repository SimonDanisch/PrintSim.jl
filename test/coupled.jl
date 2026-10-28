using PrintSim, Test, LinearAlgebra
import PrintSim as PS
using GeometryBasics: Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

function coupled_momentum(backend; steps = 12, kw...)
    b = ModelBuilder(; gravity = Vec3f(0))
    axes = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))
    add_body!(TEST_DEVICE, b, box_mesh(Vec3(0.0, 0, 0), axes, (0.02, 0.02, 0.002)); mass = 0.002, static = 0.5, kinetic = 0.4)
    add_sheet!(b, SheetGrid(0.02, 0.02, 9, 9), SheetMaterial();
        place = p -> Vec3(p[1] - 0.01, p[2] - 0.01, 0.002 + 0.5e-4 + 4e-6))
    b.v .= Ref(Vec3f(0.02, 0, 0))
    state = WorldState(backend, b)
    solver = WorldSolver(backend, b; kw...)
    PS.prime!(solver, state)
    precise(p) = Vec3{Float64}(p.hi) + Vec3{Float64}(p.lo)
    momentum() = sum(b.mass[i] * precise(v) for (i, v) in enumerate(Array(state.sheets.v))) +
        b.bodies[1].mass[1] * precise(Array(state.bodies.v)[1])
    before = momentum()
    for _ in 1:steps
        PS.step!(solver, state, 1e-3)
    end
    return before, momentum(), solver, state, b
end

@testset "sheet–body contacts conserve momentum" begin
    before, after, solver, state, b = coupled_momentum(BACKEND)
    @test norm(after - before) < 1e-4 * norm(before)
    @test all(isfinite, after)
    @test all(iszero, Array(solver.bodies.contacts.overflow))
end
