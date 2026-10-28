using PrintSim, Test, LinearAlgebra
import PrintSim as PS
import KernelInterface as KI
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

function accd_cases!(out, x0, x1, kinds)
    i = KI.get_global_id(Int32).x
    i <= length(out) || return nothing
    @inbounds begin
        a, b = x0[i], x1[i]
        r = kinds[i] == 0x01 ? PS.additive_ccd(a, b, Val(true), 0.0f0, Int32(4096)) :
            PS.additive_ccd(a, b, Val(false), 0.0f0, Int32(4096))
        out[i] = (r.hit, r.time, r.resolved)
    end
    return nothing
end

@testset "additive CCD on the device against exact impact times" begin
    V = PS.Vec3f
    # (start, finish, vertex/face?, exact time of impact or nothing)
    cases = [
        # Point 1 mm above a triangle, falling 2 mm: impact at t = 1/2.
        ((V(0.2e-3, 0.2e-3, 1e-3), V(0, 0, 0), V(1e-3, 0, 0), V(0, 1e-3, 0)),
         (V(0.2e-3, 0.2e-3, -1e-3), V(0, 0, 0), V(1e-3, 0, 0), V(0, 1e-3, 0)), true, 0.5),
        # The same point falling only 0.5 mm: no impact.
        ((V(0.2e-3, 0.2e-3, 1e-3), V(0, 0, 0), V(1e-3, 0, 0), V(0, 1e-3, 0)),
         (V(0.2e-3, 0.2e-3, 0.5e-3), V(0, 0, 0), V(1e-3, 0, 0), V(0, 1e-3, 0)), true, nothing),
        # Crossing edges 0.4 mm apart in z, the upper one descending 1 mm: impact at t = 0.4.
        ((V(-1e-3, 0, 0.4e-3), V(1e-3, 0, 0.4e-3), V(0, -1e-3, 0), V(0, 1e-3, 0)),
         (V(-1e-3, 0, -0.6e-3), V(1e-3, 0, -0.6e-3), V(0, -1e-3, 0), V(0, 1e-3, 0)), false, 0.4),
        # A 43 mm sliver 65 µm from a point that approaches by 30 µm: no impact.
        ((V(1e-3, 20e-3, 65e-6), V(0, 0, 0), V(0, 43e-3, 0), V(0.05e-3, 43e-3, 0)),
         (V(1e-3, 20e-3, 35e-6), V(0, 0, 0), V(0, 43e-3, 0), V(0.05e-3, 43e-3, 0)), true, nothing),
    ]
    x0 = PS.upload(BACKEND, [c[1] for c in cases])
    x1 = PS.upload(BACKEND, [c[2] for c in cases])
    kinds = PS.upload(BACKEND, UInt8[c[3] for c in cases])
    out = PS.upload(BACKEND, fill((false, 0.0f0, false), length(cases)))
    PS.launch!(BACKEND, accd_cases!, (out, x0, x1, kinds), length(cases))
    for ((_, _, _, exact), (hit, time, resolved)) in zip(cases, Array(out))
        @test resolved
        if exact === nothing
            @test !hit && time == 1
        else
            # A safe time: never after the impact, and within a tenth of the gap of it.
            @test hit && 0.85exact <= time <= exact
        end
    end
end
