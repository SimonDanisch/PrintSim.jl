using PrintSim,Test,LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select an explicit device with native Float64")

@testset "FP64 elastic Krylov action retains the relative-coordinate adjoint" begin
    b=ModelBuilder(;gravity=PS.Vec3f(0))
    points,tets=tet_box((.002,.002,.002),(2,2,2);origin=PS.Vec3d(.1,0,0))
    add_deformable!(b,points,tets;young=1.46e9)
    f=PS.FEMSolver(BACKEND,b;krylov_precision=:float64,compact_gradients=true)
    s=PS.FEMState(BACKEND,b)
    PS.fem_linearize!(BACKEND,f,s)
    n=f.model.nodes
    direction=[PS.Vec3d(1e-5+sin(i)*1e-10,-2e-5+cos(i)*1e-10,3e-5+sin(2i)*1e-10) for i in 1:n]
    copyto!(f.p,direction)
    PS.fem_product!(BACKEND,f,Float32(.001))
    actual=Array(f.q)
    model=PS.FEMModel(Array,b)
    gradients=Array(f.gradients)
    coefficient=Float64.(model.mass)./Float64(Float32(.001))^2
    reference=[coefficient[i]*direction[i] for i in 1:n]
    for e in 1:model.elements,c in 1:7
        first=model.tets[1,e]
        motion=sum(dot(PS.Vec3d(gradients[c,j,e]),direction[model.tets[j,e]]-direction[first]) for j in 2:10)
        weight=Float64(model.stiffness[c,e])*motion
        for j in 2:10
            value=weight*PS.Vec3d(gradients[c,j,e])
            reference[model.tets[j,e]]+=value
            reference[first]-=value
        end
    end
    @test eltype(f.p)===PS.Vec3d
    @test eltype(f.force)===PS.Vec3d
    @test actual≈reference rtol=1e-10 atol=1e-13
    elastic=[actual[i]-coefficient[i]*direction[i] for i in 1:n]
    @test norm(sum(elastic))<=1e-11*max(sum(norm,elastic),1e-20)
    @test all(p->all(isfinite,p),actual)
    @test_throws ArgumentError PS.FEMSolver(BACKEND,b;krylov_precision=:unknown)
    shift=PS.Position(PS.Vec3d(.1,0,0))
    advanced=PS.moved(shift,PS.Vec3d(1e-15,0,0),1f0)
    increment=(Float64(advanced.hi[1])-Float64(shift.hi[1]))+
        (Float64(advanced.lo[1])-Float64(shift.lo[1]))
    @test abs(increment-1e-15)<=2eps(shift.lo[1])
end

@testset "FP64 mapped solid contact retains wide Krylov workspace" begin
    b=ModelBuilder()
    axes=(PS.Vec3d(1,0,0),PS.Vec3d(0,1,0),PS.Vec3d(0,0,1))
    add_mesh!(b,box_mesh(PS.Vec3d(0,0,-.005),axes,(.1,.1,.005));static=.5)
    points,tets=tet_box((.01,.01,.003),(2,2,2))
    gap=resting_gap(1000*.003*9.81)
    add_deformable!(b,points,tets;density=1000,young=1e6,place=p->p+PS.Vec3d(0,0,gap))
    w=WorldSolver(BACKEND,b;newton=6,iterations=128,cg_tolerance=.01,newton_tolerance=1e-8,
        surface_maps=Dict{Int,Any}(),elastic_krylov_precision=:float64,
        compact_gradients=true,matrix_free_blocks=true)
    s=WorldState(BACKEND,b)
    @test eltype(w.fem_contacts.workspace.direction)===PS.Vec3d
    @test eltype(w.fem_contacts.workspace.force)===PS.Vec3d
    @test eltype(w.fem_contacts.workspace.diagonal)===PS.Mat{3,3,Float64,9}
    PS.prime!(w,s)
    history=PS.upload(BACKEND,zeros(Float64,4,w.newton))
    stepper=Stepper(w,s,.001,1;reusable_newton=true,split_newton=true,solve_history=history)
    PS.run!(stepper);PS.waitfor!(stepper)
    @test maximum(Array(w.bodies.contacts.lagged))>0
    @test all(iszero,Array(w.fem_contacts.overflow))
    @test all(p->all(isfinite,p.hi)&&all(isfinite,p.lo),Array(s.deformables.x))
end

@testset "mapped flat boundary may contain fewer vertices than P2 nodes" begin
    b=ModelBuilder(;gravity=PS.Vec3f(0))
    points,tets=tet_box((.002,.002,.002),(2,2,2))
    part=add_deformable!(b,points,tets)
    piece=b.deformables[part]
    proxy=PS.quadratic_surface_proxy(piece.x,PS.quadratic_boundary_faces(piece);
        minimum_rounds=0,tolerance=1e-9)
    @test length(proxy.points)<length(piece.surface_nodes)
    maps=PS.install_surface_proxies!(b,Dict(part=>proxy))
    @test_throws ArgumentError FEMSolver(BACKEND,b;krylov_precision=:float64)
    solver=FEMSolver(BACKEND,b;surface_maps=maps,compact_gradients=true,krylov_precision=:float64)
    state=FEMState(BACKEND,b)
    PS.fem_linearize!(BACKEND,solver,state)
    @test solver.model isa PS.MappedFEMModel
    @test all(isfinite,reduce(vcat,Array(solver.force)))
end

@testset "FP64 elastic vectors compose with coupled recorded Newton" begin
    b=ModelBuilder(;gravity=PS.Vec3f(0,0,-9.81))
    points,tets=tet_box((.002,.002,.002),(2,2,2);origin=PS.Vec3d(.1,0,0))
    add_deformable!(b,points,tets;young=1e5,fixed=p->p[1]<=.100001)
    world=WorldSolver(BACKEND,b;newton=6,iterations=32,cg_tolerance=1e-3,
        newton_tolerance=1e-8,elastic_krylov_precision=:float64)
    state=WorldState(BACKEND,b)
    @test eltype(world.energy.cg_scalars)===Float64
    @test eltype(world.energy.cg_partial)===Float64
    PS.prime!(world,state)
    history=PS.upload(BACKEND,zeros(Float64,4,6))
    st=Stepper(world,state,.001,1;reusable_newton=true,split_newton=true,solve_history=history)
    PS.run!(st);PS.waitfor!(st)
    @test all(p->all(isfinite,p.hi) && all(isfinite,p.lo),Array(state.deformables.x))
end

@testset "FP64 two-pass CG reduction retains cancellation-sensitive products" begin
    a=PS.upload(BACKEND,fill(PS.Vec3d(1,0,0),512))
    b=PS.upload(BACKEND,[PS.Vec3d(isodd(i) ? 1.0 : -1.0 + 2.0^-40,0,0) for i in 1:512])
    partial=PS.upload(BACKEND,zeros(Float64,2))
    scalar=PS.upload(BACKEND,zeros(Float64,1))
    PS.launch!(BACKEND,PS.dot_into!,(partial,a,b,Int32(512),Int32(1),Val(1)),512)
    PS.launch!(BACKEND,PS.dot_into!,(scalar,partial,nothing,Int32(2),Int32(1),Val(2)),256)
    @test only(Array(scalar))==2.0^-32
end
