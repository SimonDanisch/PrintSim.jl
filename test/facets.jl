# Standalone convergence study: julia --project=. dev/PrintSim/test/facets.jl
using PrintSim, LinearAlgebra
import PrintSim as PS
import GeometryBasics as GB
using GeometryBasics: Vec3, Point3d
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

"Smallest sheet-surface clearance at sampled upper cylinder edges (m)."
function roller_clearance(st, solver, R, N; samples = 80)
    x = Array(st.x)
    tris = Array(solver.model.tris)
    r = solver.radius
    clearance = Inf
    for k in 0:(N - 1)
        φ = 2π * k / N
        px, pz = R * sin(φ), R * cos(φ)
        pz < R * cos(π / 3) && continue
        for y in range(-0.024, 0.024; length = samples)
            q = PS.Position(Vec3(px, y, pz))
            for t in tris
                g, _, _ = PS.triangle_gap(q, x[t[1]], x[t[2]], x[t[3]], r, 1.0f0, 1.0f0)
                clearance = min(clearance, abs(g + r) - r)
            end
        end
    end
    return clearance
end

"A 50 mm sheet settling on a 12 mm roller for 600 steps of 1 ms."
function facet_run(backend, N, stiffness; steps = 600, n = 17, size = 0.05, R = 0.006, dt = 1e-3)
    b = ModelBuilder()
    roller = GB.mesh(GB.Tessellation(GB.Cylinder(Point3d(0, -0.03, 0), Point3d(0, 0.03, 0), R), 2N); pointtype = Point3d)
    add_mesh!(b, roller; static = 0.5, kinetic = 0.4, stiffness)
    material = SheetMaterial()
    add_sheet!(b, SheetGrid(size, size, n, n), material;
        place = p -> Vec3(p[1] - size / 2, p[2] - size / 2, R + 0.5e-4 + 1.0e-5))
    solver = SheetSolver(backend, b)
    st = State(backend, b)
    PS.prime!(solver, st)
    weight = material.grammage * size^2 * 9.81
    load = Float64[]
    for _ in 1:steps
        PS.step!(solver, st, dt)
        push!(load, (sum(Array(solver.touches.lagged)) + sum(Array(solver.faces.lagged))) / weight)
    end
    x = approximate.(Array(st.x))
    return (; N, n, stiffness, sagitta_um = 1e6 * R * (1 - cos(π / N)),
        early_load = extrema(load[100:300]), late_load = extrema(load[300:end]),
        final_load = load[end], final_range = extrema(load[(end - 99):end]),
        clearance_um = 1e6 * roller_clearance(st, solver, R, N),
        lowest_mm = 1e3 * minimum(p -> p[3], x))
end

if abspath(PROGRAM_FILE) == @__FILE__
    for n in (17, 33), stiffness in (2e9, 2e10), N in (32, 64, 128, 256)
        println(facet_run(BACKEND, N, stiffness; n))
        flush(stdout)
    end
end
