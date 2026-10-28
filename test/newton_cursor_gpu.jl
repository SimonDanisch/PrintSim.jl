using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select a coordinated hardware or explicit software backend")

@testset "resident split Newton continuation matches uninterrupted resident plans" begin
    b=ModelBuilder(;gravity=PS.Vec3f(0,0,-9.81))
    add_sheet!(b,SheetGrid(.01,.01,3,3),SheetMaterial())
    points,tets=tet_box((.002,.002,.002),(2,2,2);origin=PS.Vec3d(.1,0,0))
    add_deformable!(b,points,tets;young=1e5,fixed=p->p[1]<=.100001)
    solvers=[WorldSolver(BACKEND,b;newton=4,iterations=24,cg_tolerance=1e-3,
        newton_tolerance=1e-8) for _ in 1:2]
    states=[WorldState(BACKEND,b) for _ in 1:2]
    histories=[PS.upload(BACKEND,zeros(Float32,4,4)) for _ in 1:2]
    predictions=zeros(Int,2)
    audits=[Int[],Int[]]
    steppers=Base.map(1:2) do i
        PS.prime!(solvers[i],states[i])
        Stepper(solvers[i],states[i],.001,2;reusable_newton=true,split_newton=true,
            solve_history=histories[i],stage_deadline=i==1 ? Inf : 0.,
            on_prediction=()->(predictions[i]+=1),on_step=step->push!(audits[i],step))
    end
    PS.run!(steppers[1]);PS.waitfor!(steppers[1])
    cursor=PS.ReusableNewtonCursor(steppers[2])
    receipts=NamedTuple[]
    while cursor.phase!==:done
        expired=PS.advance_reusable_newton!(cursor;deadline=time()-1)
        @test expired.executed_stages==0
        push!(receipts,PS.advance_reusable_newton!(cursor;deadline=time()+120,max_stages=1))
        @test last(receipts).executed_stages==1
        @test length(audits[2])==last(receipts).completed_steps
    end
    @test predictions==[2,2]
    @test audits[1]==audits[2]==[1,2]
    @test last(receipts).complete
    @test !cursor.poisoned
    precise(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
    for field in (:sheets,:bodies,:deformables)
        a,c=getproperty(states[1],field),getproperty(states[2],field)
        coordinate=field===:bodies ? :q : :x
        @test precise.(Array(getproperty(a,coordinate))) ≈ precise.(Array(getproperty(c,coordinate))) atol=1e-12 rtol=1e-10
        velocity(p)=p isa PS.Position ? precise(p) : PS.Vec3d(p)
        @test velocity.(Array(a.v)) ≈ velocity.(Array(c.v)) atol=1e-7 rtol=1e-5
    end
    @test Array(histories[1]) ≈ Array(histories[2]) atol=1e-10 rtol=1e-5
    @test Array(solvers[1].energy.newton_status)==Array(solvers[2].energy.newton_status)
end
