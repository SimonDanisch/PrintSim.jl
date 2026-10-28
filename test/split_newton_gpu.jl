using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select an explicit software or coordinated GPU backend")

@testset "synchronized linear/trial split preserves CCD, Newton and replay" begin
    b=ModelBuilder(;gravity=PS.Vec3f(0,0,-9.81))
    add_sheet!(b,SheetGrid(.01,.01,3,3),SheetMaterial())
    points,tets=tet_box((.002,.002,.002),(2,2,2);origin=PS.Vec3d(.1,0,0))
    add_deformable!(b,points,tets;young=1e5,fixed=p->p[1]<=.100001)
    worlds=[WorldSolver(BACKEND,b;newton=4,iterations=24,cg_tolerance=.001,
        newton_tolerance=1e-8,continuous_collisions=true,ccd_query_tile=3,
        ccd_pair_capacity=512,ccd_pair_batch=7,ccd_max_iterations=512) for _ in 1:2]
    states=[WorldState(BACKEND,b) for _ in worlds]
    histories=[PS.upload(BACKEND,zeros(Float32,4,4)) for _ in worlds]
    linear_calls=Int[]
    prediction_calls=Ref(0)
    steppers=map(eachindex(worlds)) do i
        PS.prime!(worlds[i],states[i])
        println((split_newton=i==2,stage=:primed));flush(stdout)
        Stepper(worlds[i],states[i],.001,1;reusable_newton=true,
            split_newton=i==2,solve_history=histories[i],
            on_linear=i==2 ? (n->push!(linear_calls,n)) : nothing,
            on_prediction=i==2 ? (()->(prediction_calls[]+=1)) : nothing)
    end
    for replay in 1:2
        empty!(linear_calls)
        for st in steppers;PS.run!(st);PS.waitfor!(st);end
        @test linear_calls==collect(1:Int(Array(worlds[2].energy.newton_status)[2]))
        @test prediction_calls[]==replay
        @test Array(histories[1])≈Array(histories[2]) rtol=1e-6 atol=1e-10
        @test Array(worlds[1].energy.newton_status)==Array(worlds[2].energy.newton_status)
        @test Array(worlds[1].ccd.status)==Array(worlds[2].ccd.status)
        @test only(Array(worlds[2].ccd.status)) & UInt32(2)==0
        @test only(Array(worlds[2].ccd_pairs.state.running))==0
        @test Array(worlds[1].ccd.bound)==Array(worlds[2].ccd.bound)
        for field in (:sheets,:bodies,:deformables)
            a,c=getproperty(states[1],field),getproperty(states[2],field)
            name=field===:bodies ? :q : :x
            precise(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
            @test precise.(Array(getproperty(a,name)))≈precise.(Array(getproperty(c,name))) atol=1e-12 rtol=1e-10
        end
    end
    @test_throws ArgumentError Stepper(worlds[1],states[1],.001,1;split_newton=true)
    @test_throws ArgumentError Stepper(worlds[1],states[1],.001,1;reusable_newton=true,on_linear=identity)
end
