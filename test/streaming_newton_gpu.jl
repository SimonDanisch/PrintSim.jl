using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select an explicit software or coordinated GPU backend")

@testset "streaming CCD composes with sparse mapped contacts and reusable Newton" begin
    b=ModelBuilder(;gravity=PS.Vec3f(0))
    add_sheet!(b,SheetGrid(.01,.01,3,3),SheetMaterial())
    points,tets=tet_box((.002,.002,.002),(2,2,2);origin=PS.Vec3d(.1,0,0))
    add_deformable!(b,points,tets;young=1e5,fixed=p->true)
    worlds=[WorldSolver(BACKEND,b;newton=4,iterations=8,newton_tolerance=1e-8,
        continuous_collisions=true,surface_maps=Dict{Int,Any}(),
        ccd_pair_capacity=512,ccd_pair_batch=7,ccd_query_tile=tile) for tile in (0,3)]
    states=[WorldState(BACKEND,b) for _ in worlds]
    histories=[PS.upload(BACKEND,zeros(Float32,4,4)) for _ in worlds]
    for (w,s) in zip(worlds,states);PS.prime!(w,s);end
    steppers=[Stepper(w,s,.001,1;reusable_newton=true,solve_history=h)
        for (w,s,h) in zip(worlds,states,histories)]
    for replay in 1:2
        for st in steppers;PS.run!(st);PS.waitfor!(st);end
        for field in (:sheets,:deformables)
            precise(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
            a,c=getproperty(states[1],field),getproperty(states[2],field)
            @test precise.(Array(a.x)) ≈ precise.(Array(c.x)) atol=1e-12 rtol=1e-10
        end
        @test Array(histories[1])==Array(histories[2])
        @test Array(worlds[1].energy.newton_status)==Array(worlds[2].energy.newton_status)
        @test Array(worlds[1].ccd.status)==Array(worlds[2].ccd.status)==UInt32[0]
        @test Array(worlds[1].ccd.bound)==Array(worlds[2].ccd.bound)
        for (world,history) in zip(worlds,histories)
            @test all(iszero,Array(world.fem_contacts.overflow))
        end
    end
end
