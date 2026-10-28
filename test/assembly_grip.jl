using PrintSim,Test,LinearAlgebra
import PrintSim as PS

@testset "bounded nodal assembly actuator force law" begin
    for delta in (PS.Vec3f(0),PS.Vec3f(1e-8,2e-8,-1e-8),
            PS.Vec3f(.1,-.2,.3),PS.Vec3f(1e30,2e30,-1e30)),w in (0.1f0,1f0)
        f=PS.assembly_grip_force(delta,w,1f5,3f0)
        @test all(isfinite,f)
        @test norm(f)<=w*3f0*(1+5eps(Float32))
        @test dot(Float64.(f),Float64.(delta))>=0
        @test PS.assembly_grip_force(-delta,w,1f5,3f0)==-f
    end
    @test PS.assembly_grip_force(PS.Vec3f(1e-6,0,0),1f0,100f0,2f0)≈PS.Vec3f(1e-4,0,0)
    @test (@inferred PS.assembly_grip_force(PS.Vec3f(1),1f0,100f0,2f0)) isa PS.Vec3f
end
