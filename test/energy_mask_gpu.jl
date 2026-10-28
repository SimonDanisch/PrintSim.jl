using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select an explicit software or coordinated GPU backend")

@testset "constant device energy mask preserves dense dot reduction" begin
    for n in (0,1,257,4096), nonfinite in (false,true)
        values=[PS.Vec3f(mod(i,17),mod(i,3),-mod(i,5)) for i in 1:max(n,1)]
        nonfinite && n>0 && (values[end]=PS.Vec3f(1,Inf,NaN))
        a=PS.upload(BACKEND,values)
        dense=PS.upload(BACKEND,fill(PS.Vec3f(1,0,0),max(n,1)))
        mask=PS.RepeatedEnergyMask(PS.upload(BACKEND,[PS.Vec3f(1,0,0)]))
        results=[PS.upload(BACKEND,Float32[0]) for _ in 1:2]
        for (out,b) in zip(results,(dense,mask))
            PS.launch!(BACKEND,PS.dot_into!,
                (out,a,b,Int32(n),Int32(1)),max(256,cld(n,256)*256))
        end
        @test isequal(Array(results[1]),Array(results[2]))
        @test length(mask.data)==1
    end
    b=ModelBuilder()
    add_sheet!(b,SheetGrid(.01,.01,3,3),SheetMaterial())
    @test length(PS.WorldEnergy(BACKEND,b).ones)==1
    axes=(PS.Vec3d(1,0,0),PS.Vec3d(0,1,0),PS.Vec3d(0,0,1))
    add_body!(TEST_DEVICE, b,box_mesh(PS.Vec3d(.1,0,0),axes,(.01,.01,.01)))
    @test length(PS.BodySolver(BACKEND,b;allow_empty=true).energy_ones)==1
end
