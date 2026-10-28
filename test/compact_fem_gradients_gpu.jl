using PrintSim,Test,LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select an explicit software or coordinated GPU backend")

@testset "matrix-free invariant gradients preserve force, tangent and Schwarz" begin
    b=ModelBuilder()
    points,tets=tet_box((.006,.004,.002),(3,2,2))
    add_deformable!(b,points,tets;young=1e5,fixed=p->p[1]==0)
    solvers=[FEMSolver(BACKEND,b;schwarz=true,compact_gradients=compact,matrix_free_blocks=compact) for compact in (false,true)]
    states=[FEMState(BACKEND,b) for _ in solvers]
    @test size(solvers[2].gradients)==size(solvers[1].gradients)
    @test length(solvers[2].gradients.deformation)*70==length(solvers[1].gradients)*3
    for amplitude in (0.,1e-6,1e-5)
        x=PS.Position.([p+PS.Vec3d(amplitude*sin(700p[2]),amplitude*cos(900p[1]),amplitude*sin(800p[1]))
            for p in only(b.deformables).initial])
        for (s,state) in zip(solvers,states)
            copyto!(state.x,x);copyto!(s.inertia,state.x)
            PS.fem_linearize!(BACKEND,s,state)
            PS.fem_rhs!(BACKEND,s,state,1f-3)
            PS.fem_prepare_preconditioner!(BACKEND,s)
        end
        for field in (:force,:block,:strain,:r,:inverse)
            a,c=Array(getproperty(solvers[1],field)),Array(getproperty(solvers[2],field))
            @test norm(a-c)<=1e-7+3e-6*norm(a)
        end
        a,c=Array(solvers[1].gradients),Array(solvers[2].gradients)
        @test norm(a-c)<=1e-5+3e-6*norm(a)
        @test Array(solvers[1].schwarz.fallback)==Array(solvers[2].schwarz.fallback)
        @test Array(solvers[1].schwarz.factor)≈Array(solvers[2].schwarz.factor) rtol=5e-4 atol=1e-6
        residual=PS.Vec3f[1e-6*PS.Vec3f(sin(i),cos(i/3),sin(i/7)) for i in 1:solvers[1].model.nodes]
        for s in solvers
            copyto!(s.p,residual);PS.fem_product!(BACKEND,s,1f-3)
            copyto!(s.r,residual)
            PS.launch!(BACKEND,PS.fem_precondition!,
                (s.z,s.r,s.inverse,s.model),s.model.nodes)
            PS.launch!(BACKEND,PS.schwarz_apply!,
                (s.z,s.r,s.schwarz,s.model),s.schwarz.patches)
        end
        for field in (:q,:z)
            a,c=Array(getproperty(solvers[1],field)),Array(getproperty(solvers[2],field))
            @test norm(a-c)<=1e-10+5e-5*norm(a)
            @test all(isfinite,c)
        end
    end
end

@testset "compact gradients preserve complete coupled Newton replay" begin
    b=ModelBuilder()
    points,tets=tet_box((.002,.002,.002),(2,2,2))
    add_deformable!(b,points,tets;young=1e5,fixed=p->p[1]==0)
    add_sheet!(b,SheetGrid(.01,.01,3,3),SheetMaterial();place=p->p+PS.Vec3d(.1,0,0))
    worlds=[WorldSolver(BACKEND,b;newton=4,iterations=24,cg_tolerance=.001,
        newton_tolerance=1e-8,elastic_schwarz=true,compact_gradients=compact,matrix_free_blocks=compact) for compact in (false,true)]
    states=[WorldState(BACKEND,b) for _ in worlds]
    histories=[PS.upload(BACKEND,zeros(Float32,4,4)) for _ in worlds]
    steppers=map(eachindex(worlds)) do i
        PS.prime!(worlds[i],states[i])
        Stepper(worlds[i],states[i],.001,1;reusable_newton=true,split_newton=true,
            solve_history=histories[i])
    end
    for _ in 1:2
        for st in steppers;PS.run!(st);PS.waitfor!(st);end
        precise(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
        @test precise.(Array(states[1].deformables.x))≈precise.(Array(states[2].deformables.x)) rtol=1e-5 atol=1e-10
        @test precise.(Array(states[1].sheets.x))≈precise.(Array(states[2].sheets.x)) rtol=1e-5 atol=1e-10
        for (w,h) in zip(worlds,histories)
            @test all(isfinite,Array(w.energy.sum))
        end
    end
end
