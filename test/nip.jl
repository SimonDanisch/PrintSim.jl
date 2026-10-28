# A mechanically free motor-driven roller, two sheets and a force-loaded pad. This is a
# coupled-solver validation fixture, not a substitute for the feeder's complete assembly.
using PrintSim, Test, LinearAlgebra
import PrintSim as PS
import GeometryBasics as GB
using GeometryBasics: Point3d, Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include(joinpath(@__DIR__, "backend.jl"))

function driven_nip(backend; steps = 160, iterations = 64, newton = 8, diagnostics = false, recorded = true, batch = 20)
    R, width, L, dt = 0.012, 0.030, 0.050, 0.001
    b = ModelBuilder(; gravity = Vec3f(0))
    roller = GB.mesh(GB.Tessellation(GB.Cylinder(Point3d(-width / 2, 0, -R), Point3d(width / 2, 0, -R), R), 1024); pointtype = Point3d)
    rotor = add_body!(TEST_DEVICE, b, roller; mass = 0.02, static = 1.2, kinetic = 1.0)
    add_joint!(b, rotor; position = Vec3(0.0, 0, -R), axis = Vec3(1.0, 0, 0))
    motor = add_stepper!(b, rotor; torque = 0.0373)
    axes = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))
    pad = add_body!(TEST_DEVICE, b, box_mesh(Vec3(0.0, 0, 0.0012), axes, (width / 2, 0.001, 0.001));
        mass = 0.01, static = 0.8, kinetic = 0.6)
    add_joint!(b, pad; position = Vec3(0.0, 0, 0.0012), axis = Vec3(0.0, 0, 1), kind = :prismatic)
    sheets = [add_sheet!(b, SheetGrid(L, width, 33, 17), SheetMaterial();
        place = p -> Vec3(p[2] - width / 2, p[1] - L / 2, (layer - 0.5) * 1e-4 + layer * 4e-6)) for layer in 1:2]
    stack!(b, sheets)
    solver, state = WorldSolver(backend, b; iterations, newton), WorldState(backend, b)
    PS.prime!(solver, state)
    set_load!(solver, pad; force = Vec3f(0, 0, -0.9))
    command!(solver, motor; rpm = 6.5, steps = -1000)
    rec = Recording(b; names = ["driven roller", "separation pad"], colors = [:darkorange, :burlywood])
    record!(rec, solver, state, 0.0)
    before = [sum(b.x[i][2] for i in vertices(sheet)) / length(vertices(sheet)) for sheet in sheets]
    leading(sheet) = [vertex(sheet, sheet.grid.nu, j) for j in 1:sheet.grid.nv]
    nipline(sheet) = [vertex(sheet, (sheet.grid.nu + 1) ÷ 2, j) for j in 1:sheet.grid.nv]
    leading_before = [sum(b.x[i][2] for i in leading(sheet)) / sheet.grid.nv for sheet in sheets]
    nip_before = [sum(b.x[i][2] for i in nipline(sheet)) / sheet.grid.nv for sheet in sheets]
    stepper = recorded ? Stepper(solver, state, dt, batch) : nothing
    diagnostics && println("Nip solver ready; starting timesteps.")
    flush(stdout)
    for first in 1:batch:steps
        count = min(batch, steps - first + 1)
        if recorded && count == batch
            PS.run!(stepper)
            waitfor!(stepper)
        else
            for _ in 1:count
                PS.step!(solver, state, dt)
            end
        end
        step = first + count - 1
        if step % batch == 0
            record!(rec, solver, state, step * dt)
            diagnostics && (step <= 5 || step % 20 == 0) && println((step, force = sum(Array(solver.sheets.touches.lagged)) + sum(Array(solver.sheets.faces.lagged)),
                angle = Array(solver.bodies.motor_state.angle), commanded = Array(solver.bodies.motor_state.commanded), residual = norm(Array(solver.sheets.mg.levels[1].r)),
                backtrack = reinterpret(Float32, Array(solver.energy.bound)[1]), accepted = Array(solver.energy.accepted), cg = Array(solver.energy.cg_iterations)))
            flush(stdout)
        end
    end
    x = Array(state.sheets.x)
    precise(p) = Vec3{Float64}(p.hi) + Vec3{Float64}(p.lo)
    mean_travel = [sum(precise(x[i])[2] for i in vertices(sheet)) / length(vertices(sheet)) - before[k] for (k, sheet) in enumerate(sheets)]
    travel = [sum(precise(x[i])[2] for i in leading(sheet)) / sheet.grid.nv - leading_before[k] for (k, sheet) in enumerate(sheets)]
    nip_travel = [sum(precise(x[i])[2] for i in nipline(sheet)) / sheet.grid.nv - nip_before[k] for (k, sheet) in enumerate(sheets)]
    zrange = [extrema(precise(x[i])[3] for i in vertices(sheet)) for sheet in sheets]
    return (; travel, mean_travel, nip_travel, zrange, rec, solver, state)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = driven_nip(BACKEND; diagnostics = true)
    @show result.travel
end
