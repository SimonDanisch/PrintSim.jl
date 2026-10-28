using PrintSim, Test, LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

@testset "element-wise FEM product and bounded RHS preserve the operator" begin
    b = ModelBuilder()
    points, tets = tet_box((0.008, 0.006, 0.002), (4, 3, 3))
    add_deformable!(b, points, tets; fixed = p -> p[1] == 0,
        place = p -> p + PS.Vec3d(1e-6 * sin(1000p[2]), 0, 0))
    solver = FEMSolver(BACKEND, b; iterations = 16)
    state = FEMState(BACKEND, b)
    model, dt = solver.model, 1.0f-3
    n = model.nodes
    PS.launch!(BACKEND, PS.fem_predict!, (state, solver.anchor, solver.inertia, model, solver.one, solver.gravity, dt), n)
    PS.launch!(BACKEND, PS.fem_linearize!, (solver.force, solver.gradients, solver.block, solver.strain, state, model, solver.one), model.elements)
    reference_force, reference_block = Array(solver.force), Array(solver.block)
    for offset in 0:13:(model.elements - 1)
        count = min(13, model.elements - offset)
        PS.launch!(BACKEND, PS.fem_linearize!, (solver.force, solver.gradients, solver.block,
            solver.strain, state, model, solver.one, Int32(offset), Int32(count)), count)
    end
    @test Array(solver.force) == reference_force
    @test Array(solver.block) == reference_block
    PS.launch!(BACKEND, PS.fem_rhs!, (solver.r, solver.inverse, state, solver.inertia,
        solver.external, solver.force, solver.block, model, dt), n)
    reference_r, reference_inverse = Array(solver.r), Array(solver.inverse)
    for offset in 0:7:(n - 1)
        count = min(7, n - offset)
        PS.launch!(BACKEND, PS.fem_rhs!, (solver.r, solver.inverse, state, solver.inertia,
            solver.external, solver.force, solver.block, model, dt, Int32(offset), Int32(count)), count)
    end
    @test Array(solver.r) == reference_r
    @test Array(solver.inverse) == reference_inverse
    copyto!(solver.p, PS.Vec3f[1e-6 * PS.Vec3f(sin(i), cos(i / 3), sin(i / 7)) for i in 1:n])
    PS.launch!(BACKEND, PS.fem_apply!, (solver.q, solver.p, solver.gradients, model, dt), n)
    reference = Array(solver.q)
    PS.fem_product!(BACKEND, solver, dt)
    actual = Array(solver.q)
    @test norm(actual - reference) < 1e-5 * norm(reference)
    @test all(isfinite, actual)
end
