using PrintSim, Test, LinearAlgebra
import PrintSim as PS
import Mantle
using GeometryBasics: Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

@testset "a later finite Newton residual cannot conceal a failed solve" begin
    active = PS.upload(BACKEND, UInt32[0])
    initial = PS.upload(BACKEND, Float32[0])
    iterations = PS.upload(BACKEND, UInt32[0, 0])
    scalars = PS.upload(BACKEND, Float32[NaN, 0, 0])
    PS.launch!(BACKEND, PS.world_cg_start!, (active, initial, iterations, scalars), 1)
    @test Array(active)[1] == 0
    copyto!(scalars, Float32[1, 1, 0])
    PS.launch!(BACKEND, PS.world_cg_start!, (active, initial, iterations, scalars), 1)
    @test Array(active)[1] == 1
    @test Array(iterations)[2] & UInt32(0x80000000) != 0
end

@testset "Newton solve diagnostics retain zero, failed and capped solves" begin
    history = PS.upload(BACKEND, zeros(Float32, 4, 3))
    initial = PS.upload(BACKEND, Float32[0])
    iterations = PS.upload(BACKEND, UInt32[0, 0])
    scalars = PS.upload(BACKEND, zeros(Float32, 5))
    PS.launch!(BACKEND, PS.world_cg_record!,
        (history, initial, iterations, scalars, Int32(1), true, true, Int32(2), 0.01f0), 1)
    copyto!(initial, Float32[NaN])
    copyto!(scalars, Float32[NaN, 0, 0, 0, 0])
    PS.launch!(BACKEND, PS.world_cg_record!,
        (history, initial, iterations, scalars, Int32(2), true, true, Int32(2), 0.01f0), 1)
    copyto!(initial, Float32[1])
    copyto!(iterations, UInt32[2, 2])
    copyto!(scalars, Float32[1, 0, 0, 0, 0.5])
    PS.launch!(BACKEND, PS.world_cg_record!,
        (history, initial, iterations, scalars, Int32(3), false, true, Int32(2), 0.01f0), 1)
    actual = Array(history)
    @test actual[:, 1] == Float32[0, 0, 0, 1]
    @test isnan(actual[1, 2]) && isnan(actual[2, 2])
    @test actual[4, 2] == 0
    @test actual[:, 3] == Float32[1, 0.5, 2, 0]
end

function cg_recording_fixture(; iterations = 24, compact = true)
    b = ModelBuilder(; gravity = Vec3f(0))
    points, tets = tet_box((0.02, 0.02, 0.004), (2, 2, 2);
        origin = Vec3(-0.01, -0.01, -0.002))
    part = add_deformable!(b, points, tets; young = 1e6, density = 1000)
    axes = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))
    body = add_body!(TEST_DEVICE, b, box_mesh(Vec3(0.0, 0, -0.003), axes, (0.005, 0.005, 0.001));
        mass = 0.01, velocity = Vec3(0.01, 0, 0))
    add_bond!(b, part, body; position = Vec3(0.0, 0, -0.002))
    add_sheet!(b, SheetGrid(0.01, 0.01, 5, 5), SheetMaterial();
        place = p -> Vec3(p[1] - 0.005, p[2] - 0.005, 0.002 + 0.5e-4 + 4e-6))
    solver = WorldSolver(BACKEND, b; iterations, newton = 2)
    state = WorldState(BACKEND, b)
    PS.prime!(solver, state)
    graph = Mantle.Graph(Mantle.todevice(BACKEND))
    history = PS.upload(BACKEND, zeros(Float32, 4, solver.newton))
    PS.step!(graph, solver, state, 1e-3; compact_cg = compact, solve_history = history)
    return (; b, solver, state, graph, history)
end

@testset "compact coupled CG recording preserves the solve" begin
    compact = cg_recording_fixture()
    expanded = cg_recording_fixture(; compact = false)
    @test length(Mantle.passes(compact.graph)) < length(Mantle.passes(expanded.graph)) ÷ 3
    longer = cg_recording_fixture(; iterations = 64)
    @test length(Mantle.passes(compact.graph)) == length(Mantle.passes(longer.graph))
    plans = map((compact, expanded)) do fixture
        plan = Mantle.Plan(fixture.graph)
        Mantle.record!(plan)
        plan
    end
    # A final discarded iteration may make the measured loop look free. Even
    # with zero recorded costs, finite-budget submissions must not merge all
    # iterations into one watchdog-sized submission on the next active run.
    measured = Mantle.Plan(compact.graph)
    measured.passcost .= 0
    ranges = Mantle.partitionranges(measured)
    parts = collect(eachindex(ranges))
    batches = Mantle.submissionbatches(measured, ranges, parts)
    for (i, range) in enumerate(ranges)
        isempty(range) && continue
        loop = measured.passes[first(range)].pass.loop
        loop === nothing && continue
        @test all(batch -> count(==(i), batch) <= 1, batches)
    end
    for _ in 1:3
        foreach(Mantle.run!, plans)
        foreach(Mantle.waitfor!, plans)
        for (a, b) in ((compact.state.sheets.x, expanded.state.sheets.x),
                (compact.state.bodies.q, expanded.state.bodies.q),
                (compact.state.deformables.x, expanded.state.deformables.x))
            precise(p) = Vec3{Float64}(p.hi) + Vec3{Float64}(p.lo)
            difference=maximum(norm.(precise.(Array(a)) .- precise.(Array(b))))
            if difference>=2e-9
                # Read diagnostics AFTER detecting the mismatch. Reading these
                # unconditionally first changed the cold-run behavior, so it
                # must not accidentally hide a synchronization defect.
                cp,ep=Array(compact.solver.energy.newton_progress),Array(expanded.solver.energy.newton_progress)
                reread=maximum(norm.(precise.(Array(a)) .- precise.(Array(b))))
                @info "compact/expanded mismatch" difference reread compact=cp expanded=ep
            end
            @test difference < 2e-9
        end
        @test Array(compact.solver.energy.cg_iterations) == Array(expanded.solver.energy.cg_iterations)
        ch, eh = Array(compact.history), Array(expanded.history)
        @test all(isfinite, ch)
        @test ch[3:4, :] == eh[3:4, :]
        @test ch[1:2, :] ≈ eh[1:2, :] rtol = 5e-5 atol = 1e-20
        @test sum(ch[3, :]) == Array(compact.solver.energy.cg_iterations)[2]
        @test ch[3, end] == Array(compact.solver.energy.cg_iterations)[1]
        @test all(ch[4, j] == Float32(ch[2, j] <= max(compact.solver.cg_tolerance^2 * ch[1, j], 1f-35))
            for j in axes(ch, 2))
    end
end
