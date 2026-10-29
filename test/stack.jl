# Two sheets, one on the other, on a plane: the lower one carries both, they stick together on
# a gentle slope and slide at their own rates on a steep one.
using PrintSim, Test, LinearAlgebra
using GeometryBasics: Vec3f, Vec3
import PrintSim as PS
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")
isdefined(@__MODULE__, :slope!) || include("slopes.jl")

"""
Two 30 mm sheets stacked on a plane tilted by `θdeg`, each at its resting gap: per step the
slide of each sheet's center down the slope (m), and the normal forces on the plane and between
the sheets per sheet weight.
"""
function two_on_slope(backend; θdeg = 0.0, steps = 20, n = 17, size = 0.03, dt = 1.0e-3, plane_static = 0.5, plane_kinetic = 0.4, kw...)
    θ = deg2rad(θdeg)
    t = 1.0e-4
    b = ModelBuilder()
    up, across, normal = slope!(b, θdeg; static = plane_static, kinetic = plane_kinetic)
    m = SheetMaterial()
    g = SheetGrid(size, size, n, n)
    p1 = m.grammage * 9.81 * cos(θ)
    under, between = resting_gap(2p1), resting_gap(p1)
    lower = add_sheet!(b, g, m; place = p -> p[1] * up + p[2] * across + normal * (t / 2 + under))
    upper = add_sheet!(b, g, m; place = p -> p[1] * up + p[2] * across + normal * (3t / 2 + under + between))
    stack!(b, [lower, upper])
    solver = SheetSolver(backend, b; kw...)
    st = State(backend, b)
    PS.prime!(solver, st)
    weight = m.grammage * size^2 * 9.81 * cos(θ)
    center(x, s) = sum(i -> Vec3{Float64}(x[i].hi) + Vec3{Float64}(x[i].lo), vertices(s)) / length(vertices(s))
    x0 = Array(st.x)
    c0 = (center(x0, lower), center(x0, upper))
    slides = Tuple{Float64, Float64}[]
    loads = Tuple{Float64, Float64}[]
    for _ in 1:steps
        PS.step!(solver, st, dt)
        x = Array(st.x)
        push!(slides, (dot(center(x, lower) - c0[1], -up), dot(center(x, upper) - c0[2], -up)))
        onplane = sum(Array(solver.touches.lagged)) / weight
        between = sum(Array(solver.partners.lagged)[1, vertices(upper)]) / weight      # the upper sheet's contact below
        push!(loads, (onplane, between))
    end
    return slides, loads
end

@testset "two sheets on a plane" begin
    backend = BACKEND
    _, loads = two_on_slope(backend)
    @test all(l -> isapprox(l[1], 2; rtol = 0.005) && isapprox(l[2], 1; rtol = 0.005), loads[5:end])
    # Both come to rest on 20°. The lower sheet carries both weights, and its
    # tangential spring is half the barrier curvature at that resting load.
    slides, _ = two_on_slope(backend; θdeg = 20.0, steps = 30)
    pressure = 2 * SheetMaterial().grammage * 9.81 * cos(deg2rad(20.0))
    _, curvature = PS.normal_force(resting_gap(pressure), 2.0e10, 1.0e-6)
    give = pressure * tan(deg2rad(20.0)) / (0.5 * curvature)
    @test isapprox(slides[end][1], give; rtol = 0.05)
    @test slides[end][2] < 10give
    @test all(k -> abs(slides[end][k] - slides[end - 10][k]) < 1.0e-10, 1:2)
    slides, _ = two_on_slope(backend; θdeg = 35.0, steps = 40)
    θ = deg2rad(35.0)
    upper = 9.81 * (sin(θ) - 0.38 * cos(θ))
    lower = 9.81 * (sin(θ) + 0.38 * cos(θ) - 0.4 * 2 * cos(θ))
    # Acceleration from the change of the per-step slide between steps 20 and 40: a single
    # step's second difference carries the friction's step-to-step noise.
    rate(which) = ((slides[40][which] - slides[39][which]) - (slides[20][which] - slides[19][which])) / (20 * 1.0e-6)
    @test isapprox(rate(2), upper; rtol = 0.005)
    @test isapprox(rate(1), lower; rtol = 0.005)
end

"""
`sheets` sheets stacked on a plane tilted by `θdeg`, each at its resting gap: after `steps` steps
the largest move of a vertex (m) and over the last ten steps, and the normal force under each
sheet but the lowest per sheet weight.
"""
function deep_stack(backend; sheets = 6, θdeg = 20.0, steps = 40, n = 17, size = 0.03, kw...)
    θ = deg2rad(θdeg)
    t = 1.0e-4
    b = ModelBuilder()
    up, across, normal = slope!(b, θdeg)
    m = SheetMaterial()
    g = SheetGrid(size, size, n, n)
    p1 = m.grammage * 9.81 * cos(θ)
    height(k) = (k - 0.5) * t + sum(j -> resting_gap((sheets - j + 1) * p1), 1:k)
    pieces = [add_sheet!(b, g, m; place = p -> p[1] * up + p[2] * across + normal * height(k)) for k in 1:sheets]
    stack!(b, pieces)
    solver = SheetSolver(backend, b; kw...)
    st = State(backend, b)
    PS.prime!(solver, st)
    x0 = Array(st.x)
    late = x0
    for k in 1:steps
        PS.step!(solver, st, 1.0e-3)
        k == steps - 10 && (late = Array(st.x))
    end
    x = Array(st.x)
    weight = m.grammage * size^2 * 9.81 * cos(θ)
    moved(a, b) = maximum(norm(Vec3{Float64}(b[i] - a[i])) for i in eachindex(b))
    loads = [sum(Array(solver.partners.lagged)[1, vertices(p)]) / weight for p in pieces[2:end]]
    return moved(x0, x), moved(late, x), loads, Array(solver.mg.lost)[1]
end

@testset "a stack of six" begin
    # Each sheet carries the ones above it, and the stack comes to rest within
    # the original 22 µm settling bound, with less than 2 nm of late motion.
    move, creep, loads, lost = deep_stack(BACKEND)
    @test move < 6 * 10 * 1.0e-6 * tan(deg2rad(20.0))
    @test creep < 2.0e-9
    @test all(isapprox.(loads, 5:-1:1; rtol = 0.005))
    @test lost == 0
end
