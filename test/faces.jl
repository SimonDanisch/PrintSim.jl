# Mesh edges pressing into a sheet's faces: a sheet on two rails narrower than its grid, both
# between grid lines, so that no vertex lies over either.
using PrintSim, Test, LinearAlgebra
using GeometryBasics: Vec3f, Vec3
import PrintSim as PS
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

"""
A 50 mm sheet (16 cells a side) on two 1 mm rails 40.6 mm apart, run for `steps` steps: the
lowest point of the sheet above the rails' tops (m), and the normal force on the rails per
sheet weight.
"""
function on_rails(backend; steps = 80, n = 17, size = 0.05, dt = 1.0e-3, kw...)
    h = size / (n - 1)
    b = ModelBuilder()
    ex, ey, ez = Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1)
    for x in (1.5h, (n - 2.5) * h)
        add_mesh!(b, box_mesh(Vec3(x, size / 2, -0.5e-3), (ex, ey, ez), (0.5e-3, 0.03, 0.5e-3)); static = 0.5, kinetic = 0.4)
    end
    m = SheetMaterial()
    add_sheet!(b, SheetGrid(size, size, n, n), m; place = p -> Vec3(p[1], p[2], 0.5e-4 + 1.0e-5))
    solver = SheetSolver(backend, b; kw...)
    st = State(backend, b)
    PS.prime!(solver, st)
    weight = m.grammage * size^2 * 9.81
    lowest, load = Float64[], Float64[]
    for _ in 1:steps
        PS.step!(solver, st, dt)
        push!(lowest, minimum(p -> approximate(p)[3], Array(st.x)))
        push!(load, (sum(Array(solver.touches.lagged)) + sum(Array(solver.faces.lagged))) / weight)
    end
    return lowest, load
end

@testset "a sheet on rails between its vertices" begin
    lowest, load = on_rails(BACKEND)
    # It sags 60 µm between the rails, its faces resting on the rails' edges; each cell over a
    # rail tips onto its inner edge, and none holds the sheet down at the outer one.
    @test all(>(-1.0e-4), lowest)
    @test isapprox(load[end], 1; rtol = 0.01)
end
