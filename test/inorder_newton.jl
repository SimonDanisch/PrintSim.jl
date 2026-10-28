using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "linear-memory scheduling matches original Newton timestep and replay" begin
    for tolerance in (0.,1e-8)
        b=ModelBuilder(;gravity=PS.Vec3f(0,0,-9.81))
        add_sheet!(b,SheetGrid(.01,.01,3,3),SheetMaterial())
        points,tets=tet_box((.002,.002,.002),(2,2,2);origin=PS.Vec3d(.1,0,0))
        add_deformable!(b,points,tets;young=1e5,fixed=p->p[1]<=.100001)
        solvers=[WorldSolver(BACKEND,b;newton=4,iterations=24,cg_tolerance=1e-3,
            newton_tolerance=tolerance) for _ in 1:2]
        states=[WorldState(BACKEND,b) for _ in 1:2]
        for (solver,state) in zip(solvers,states)
            PS.prime!(solver,state)
        end
        history=[PS.upload(BACKEND,zeros(Float32,4,4)) for _ in 1:2]
        steppers=[Stepper(solvers[i],states[i],.001,1;solve_history=history[i],
            policy=i==1 ? PS.Mantle.InOrder() : PS.Mantle.Overlap()) for i in 1:2]
        ordered,reference=(s.plan.plan for s in steppers)
        @test [p.pass.name for p in ordered.passes]==[p.pass.name for p in reference.passes]
        println((newton_layout=:inorder_vs_original,tolerance,passes=length(ordered.passes)));flush(stdout)
        for _ in 1:2
            for stepper in steppers
                PS.run!(stepper);PS.waitfor!(stepper)
            end
            for field in (:sheets,:bodies,:deformables)
                a,c=getproperty(states[1],field),getproperty(states[2],field)
                coordinates=field===:bodies ? :q : :x
                precise(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
                @test precise.(Array(getproperty(a,coordinates))) ≈
                    precise.(Array(getproperty(c,coordinates))) atol=1e-12 rtol=1e-10
                velocity(p)=p isa PS.Position ? precise(p) : PS.Vec3d(p)
                av,cv=velocity.(Array(a.v)),velocity.(Array(c.v))
                @test av ≈ cv atol=1e-7 rtol=1e-5
            end
            @test Array(history[1]) ≈ Array(history[2]) atol=1e-10 rtol=1e-5
            @test Array(solvers[1].energy.newton_status)==Array(solvers[2].energy.newton_status)
            @test Array(solvers[1].energy.newton_progress) ≈
                Array(solvers[2].energy.newton_progress) atol=1e-10 rtol=1e-5
        end
    end
end
