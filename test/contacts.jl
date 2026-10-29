# A sheet on a plane: it rests with its weight, sticks on a slope below atan(μs) and slides
# above it with g (sin θ − μk cos θ).
using PrintSim, Test, LinearAlgebra
using GeometryBasics: Vec3f, Vec3
import PrintSim as PS
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")
isdefined(@__MODULE__, :slope!) || include("slopes.jl")

"""
A 50 mm square sheet laid on a plane tilted by `θdeg` at its resting gap, run for `steps` steps;
the slide of its center down the slope (m) after each step, and the normal force on the plane
per sheet weight.
"""
function on_slope(backend; θdeg = 0.0, steps = 30, n = 33, size = 0.05, dt = 1.0e-3, μs = 0.5, μk = 0.4, recorded = false, kw...)
    θ = deg2rad(θdeg)
    r = 0.5e-4
    b = ModelBuilder()
    up, across, normal = slope!(b, θdeg; static = μs, kinetic = μk)
    m = SheetMaterial()
    gap = resting_gap(m.grammage * 9.81 * cos(θ))
    add_sheet!(b, SheetGrid(size, size, n, n), m; place = p -> p[1] * up + p[2] * across + normal * (r + gap))
    solver = SheetSolver(backend, b; kw...)
    st = State(backend, b)
    PS.prime!(solver, st)
    stepper = recorded ? Stepper(solver, st, dt, 1) : nothing
    weight = m.grammage * size^2 * 9.81 * cos(θ)
    # Accumulate in Float64: summing 1089 Float32 positions masks sub-nanometer settling.
    center(x) = sum(p -> Vec3{Float64}(p.hi) + Vec3{Float64}(p.lo), x) / length(x)
    c0 = center(Array(st.x))
    slide, load = Float64[], Float64[]
    for _ in 1:steps
        if recorded
            PS.run!(stepper)
            PS.waitfor!(stepper)
        else
            PS.step!(solver, st, dt)
        end
        push!(slide, dot(center(Array(st.x)) - c0, -up))
        push!(load, sum(Array(solver.touches.lagged)) / weight)
    end
    return slide, load
end

@testset "sheet on a plane" begin
    backend = BACKEND
    _, load = on_slope(backend)
    @test all(l -> isapprox(l, 1; rtol = 0.002), load[2:end])     # the plane carries the sheet's weight
    # Sticks, tan 20° < μs: tangential stiffness is half the barrier curvature
    # at the load's resting gap. The spring balances gravity and stays there.
    slide, _ = on_slope(backend; θdeg = 20.0)
    pressure = SheetMaterial().grammage * 9.81 * cos(deg2rad(20.0))
    _, curvature = PS.normal_force(resting_gap(pressure), 2.0e10, 1.0e-6)
    give = pressure * tan(deg2rad(20.0)) / (0.5 * curvature)
    @test isapprox(slide[end], give; rtol = 0.05)
    @test abs(slide[end] - slide[end - 10]) < 1.0e-10
    replayed, _ = on_slope(backend; θdeg = 20.0, n = 5, steps = 5, recorded = true)
    @test isapprox(replayed[end], give; rtol = 0.05)
    @test maximum(abs, diff(replayed[2:end])) < 1.0e-10
    slide, _ = on_slope(backend; θdeg = 35.0)
    θ = deg2rad(35.0)
    # Slides with μk: the acceleration from the change of the per-step slide over 20 steps.
    rate = ((slide[30] - slide[29]) - (slide[10] - slide[9])) / (20 * 1.0e-6)
    @test isapprox(rate, 9.81 * (sin(θ) - 0.4 * cos(θ)); rtol = 0.002)
end
