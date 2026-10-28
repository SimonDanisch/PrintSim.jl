using PrintSim, Test, LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

function scale_checks!(output)
    PS.KI.get_global_id(Int32).x == 1 || return nothing
    big = 1.0f20 * PS.I3
    small = 1.0f-20 * PS.I3
    @inbounds output[1] = maximum(abs, PS.preconditioner_inverse(big) * big - PS.I3)
    @inbounds output[2] = maximum(abs, PS.preconditioner_inverse(small) * small - PS.I3)
    zero_position = zero(Position)
    @inbounds output[3] = PS.stretch(zero_position, (0.0f0, 0.0f0), 1.0f0)
    @inbounds output[4] = PS.stretch(zero_position, zero_position, 1.0f0)
    a = PS.Vec3f(-5e-4, -2.5e-12, 1e-6)
    b = PS.Vec3f(5e-4, -2.5e-12, 1e-6)
    c = PS.Vec3f(5e-4, 7.5e-12, 1e-6)
    w, d, n = PS.projection(a, b, c)
    @inbounds output[5] = maximum(abs, w - PS.Vec3f(0.5, 0.25, 0.25))
    @inbounds output[6] = abs(d + 1e-6)
    @inbounds output[7] = maximum(abs, n - PS.Vec3f(0, 0, 1))
    p = Position(PS.Vec3f(0))
    gap, normal, weights = PS.triangle_gap(p, p, Position(PS.Vec3f(1e-3, 0, 0)),
        Position(PS.Vec3f(2e-3, 0, 0)), 0.0f0, 1.0f0, 1.0f0)
    @inbounds output[8] = gap
    g = PS.Vec3f(2e20, 0, 0)
    weighted = sqrt(1.0f-16) * g
    @inbounds output[9] = (weighted * weighted')[1, 1]
    # Closest-point subtraction can cancel the leading coordinates while the
    # trailing coordinates remain. Treating this as a small norm correction
    # previously produced a negative length and an enormous contact normal.
    residual = Position(PS.Vec3f(0, -8.0035534e-11, -7.4942363e-10),
        PS.Vec3f(-1.2321344e-10, 7.44104e-10, 1.3932955e-9))
    @inbounds output[10] = PS.length_pair(residual, 1.0f0)
    @inbounds output[11] = PS.stretch(residual, (0.0f0, 0.0f0), 1.0f0)
    p = Position(PS.Vec3f(-0.116, -0.017466472, 0.065622896), PS.Vec3f(-3.4610736e-9, 8.9440166e-10, -1.0927932e-9))
    a = Position(PS.Vec3f(-0.116, -0.017501919, 0.06565937), PS.Vec3f(-3.3378602e-9, 1.0571483e-10, 5.0706683e-10))
    b = Position(PS.Vec3f(-0.116, -0.017501907, 0.06565932), PS.Vec3f(-3.3378602e-9, 5.187493e-10, 2.1557565e-9))
    c = Position(PS.Vec3f(-0.116, -0.017465154, 0.06562154), PS.Vec3f(-3.3378602e-9, 1.5317525e-10, -2.5973175e-9))
    gap, normal, _ = PS.triangle_gap(p, a, b, c, 0.0f0, 1.0f0, 1.0f0)
    @inbounds output[12] = norm(normal)
    @inbounds output[13] = abs(gap)
    return nothing
end

@testset "printed-feature scales stay finite on the device" begin
    output = PS.KI.zeros(BACKEND, Float32, 13)
    PS.launch!(BACKEND, scale_checks!, (output,), 1; group = 1)
    actual = Array(output)
    @test all(isfinite, actual)
    @test maximum(abs, actual[1:2]) < 1e-6
    @test actual[3:4] == [0, 0]
    @test actual[5] < 1e-6
    @test actual[6] < 1e-11
    @test actual[7] < 1e-6
    @test actual[8] == 0
    @test actual[9] ≈ 4e24 rtol = 1e-6
    exact = norm(PS.Vec3d(0, -8.0035534f-11, -7.4942363f-10) +
        PS.Vec3d(-1.2321344f-10, 7.44104f-10, 1.3932955f-9))
    @test actual[10] ≈ exact rtol = 3e-7
    @test actual[11] ≈ exact rtol = 3e-7
    @test actual[12] ≈ 1 rtol = 3e-7
    @test actual[13] < 2e-9
end
