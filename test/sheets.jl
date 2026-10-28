# Sheets against plate and beam theory.
using PrintSim, Test, LinearAlgebra
using GeometryBasics: Vec3f, Vec3
import PrintSim as PS
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

"""
A square-ish paper cantilever: grid `n`×`n`, the first two columns held, released flat at rest.
Returns the builder, the sheet and its tip vertices.
"""
function cantilever(; n = 33, length = 0.05)
    h = length / (n - 2)
    b = ModelBuilder()
    g = SheetGrid((n - 1) * h, (n - 1) * h, h)
    s = add_sheet!(b, g, SheetMaterial())
    for j in 1:g.nv, i in 1:2
        fix!(b, vertex(s, i, j))
    end
    return b, s, [vertex(s, g.nu, j) for j in 1:g.nv], (n - 2) * h
end

"Mean tip sag (m) every `every` seconds over `T`, from the device solver."
function sag(backend; n = 33, T = 0.04, dt = 1.0e-3, every = 0.005, kw...)
    b, s, tip, _ = cantilever(; n)
    solver = SheetSolver(backend, b; kw...)
    st = State(backend, b)
    out = Float64[]
    stride = round(Int, every / dt)
    for k in 1:round(Int, T / dt)
        PS.step!(solver, st, dt)
        if k % stride == 0
            x = approximate.(Array(st.x))
            push!(out, -sum(x[i][3] for i in tip) / length(tip))
        end
    end
    return out
end

@testset "paper cantilever" begin
    m = SheetMaterial()
    q = m.grammage * 9.81
    _, _, _, L = cantilever()
    plate = q * L^4 / (8 * PS.bending_stiffness(m))
    beam = plate / (1 - m.poisson^2)
    # Statics on the device: with a 10 s step inertia vanishes. The sag settles between plate
    # (edges held straight) and beam (edges free), a few % above beam on this grid; it converges
    # toward it as the grid gets finer.
    static = last(sag(BACKEND; dt = 10.0, T = 300.0, every = 10.0, newton = 4))
    @test plate < static < 1.08 * beam
end
