using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select an explicit coordinated GPU or software backend")

@testset "reusable Newton stage preserves full timestep, audits and replay" begin
    for tolerance in (0.,1e-8)
        b=ModelBuilder(;gravity=PS.Vec3f(0,0,-9.81))
        add_sheet!(b,SheetGrid(.01,.01,3,3),SheetMaterial())
        points,tets=tet_box((.002,.002,.002),(2,2,2);origin=PS.Vec3d(.1,0,0))
        add_deformable!(b,points,tets;young=1e5,fixed=p->p[1]<=.100001)
        solvers=[WorldSolver(BACKEND,b;newton=4,iterations=24,cg_tolerance=1e-3,
            newton_tolerance=tolerance) for _ in 1:2]
        states=[WorldState(BACKEND,b) for _ in 1:2]
        println((tolerance,stage=:worlds_constructed));flush(stdout)
        for (solver,state) in zip(solvers,states);PS.prime!(solver,state);end
        history=[PS.upload(BACKEND,zeros(Float32,4,4)) for _ in 1:2]
        steppers=[Stepper(solvers[i],states[i],.001,1;solve_history=history[i],
            reusable_newton=i==1,compact_linesearch=i==1,on_prepared=phase->begin
                println((layout=i==1 ? :reusable : :unrolled,phase));flush(stdout)
            end) for i in 1:2]
        compact=steppers[1].plan
        total=sum(p->length(p.passes),(compact.predict,compact.newton,compact.finish))
        @test total<length(steppers[2].plan.plan.passes)
        println((reusable_newton=true,tolerance,total_passes=total,
            reference_passes=length(steppers[2].plan.plan.passes)));flush(stdout)
        for replay in 1:2
            for (id,stepper) in enumerate(steppers)
                println((replay,layout=id==1 ? :reusable : :unrolled,stage=:before));flush(stdout)
                PS.run!(stepper);PS.waitfor!(stepper)
                println((replay,layout=id==1 ? :reusable : :unrolled,stage=:after));flush(stdout)
            end
            precise(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
            for field in (:sheets,:bodies,:deformables)
                a,c=getproperty(states[1],field),getproperty(states[2],field)
                coordinates=field===:bodies ? :q : :x
                @test precise.(Array(getproperty(a,coordinates))) ≈
                    precise.(Array(getproperty(c,coordinates))) atol=1e-12 rtol=1e-10
                velocity(p)=p isa PS.Position ? precise(p) : PS.Vec3d(p)
                @test velocity.(Array(a.v)) ≈ velocity.(Array(c.v)) atol=1e-7 rtol=1e-5
            end
            @test Array(history[1]) ≈ Array(history[2]) atol=1e-10 rtol=1e-5
            @test Array(solvers[1].energy.newton_status)==Array(solvers[2].energy.newton_status)
            @test Array(solvers[1].energy.newton_progress) ≈
                Array(solvers[2].energy.newton_progress) atol=1e-10 rtol=1e-5
            # CCD has its own precision/overflow/replay suite and a retained
            # moving-FEM narrow-phase diagnostic. This test isolates Newton
            # recording layout, not the cost of those inclusion queries.
            @test solvers[1].ccd===solvers[2].ccd===nothing
        end
    end
end
