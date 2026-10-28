# The ray checks on cubes whose overlap and clearance are known.
using PrintSim, Test
using GeometryBasics: Rect3f, Point3f, Vec3f, uv_normal_mesh
import PrintSim as PS
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

@testset "ray checks" begin
    r = PS.Rays(BACKEND; lines = 2^14)
    cube(lo) = PS.Traced(BACKEND, uv_normal_mesh(Rect3f(Point3f(lo...), Vec3f(10, 10, 10))))
    a, b, c = cube((0, 0, 0)), cube((8, 0, 0)), cube((12, 1, 1))
    I = PS.Pose()
    @test isapprox(only(PS.shared(r, [(a, I, b, I)])), 200; rtol = 1.0e-4)   # 10 × 10 × 2 mm³
    @test isapprox(only(PS.travel(r, [(a, I, c, I)], 1)), 2; atol = 1.0e-4)   # 2 mm to go
end
