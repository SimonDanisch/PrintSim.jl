using PrintSim, Test, LinearAlgebra
import PrintSim as PS
using GeometryBasics: Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include(joinpath(@__DIR__, "backend.jl"))

@testset "glued affine–FE attachment is internal to the common solve" begin
    b = ModelBuilder(; gravity = Vec3f(0))
    points, tets = tet_box((0.002, 0.002, 0.001), (2, 2, 2); origin = Vec3(-0.001, -0.001, 0))
    part = add_deformable!(b, points, tets; young = 1e6, density = 1000)
    axes = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))
    body = add_body!(TEST_DEVICE, b, box_mesh(Vec3(0.0, 0, 0.002), axes, (0.001, 0.001, 0.001)); mass = 0.001, velocity = Vec3(0.01, 0, 0))
    for p in (Vec3(-0.0006, -0.0006, 0.001), Vec3(0.0006, -0.0006, 0.001), Vec3(0.0, 0.0006, 0.001))
        add_bond!(b, part, body; position = p)
    end
    solver, state = WorldSolver(BACKEND, b; iterations = 400, newton = 8), WorldState(BACKEND, b)
    PS.prime!(solver, state)
    precise(p) = Vec3{Float64}(p.hi) + Vec3{Float64}(p.lo)
    mass = Array(solver.deformables.model.mass)
    momentum() = b.bodies[1].mass[1] * precise(Array(state.bodies.v)[1]) + sum(mass[i] * precise(v) for (i, v) in enumerate(Array(state.deformables.v)))
    before = momentum()
    stepper = Stepper(solver, state, 0.001, 3)
    PS.run!(stepper)
    waitfor!(stepper)
    after = momentum()
    @show before after Array(solver.energy.cg_iterations)
    @test norm(after - before) < 1e-4 * norm(before)
    model = PS.Adapt.adapt(Array, solver.bonds.model)
    q, x = Array(state.bodies.q), Array(state.deformables.x)
    @test maximum(norm(PS.bond_body_point(q, model, i, 1.0f0) - PS.bond_fem_point(x, model, i, 1.0f0)) for i in 1:3) < 1e-8
    @test all(iszero, Array(solver.bodies.contacts.mesh))
    @test Array(solver.bodies.colliders.excluded)[1, 2] == 1
    @test Array(solver.energy.cg_iterations)[2] <= 8 * 400
end
