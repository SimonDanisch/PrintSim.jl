include("assembly_grip.jl")
isdefined(@__MODULE__,:BACKEND) || error("select an explicit hardware or software backend")

@testset "assembly loads are bounded, frozen for Newton, and released exactly" begin
    points=[PS.Vec3d(.119,0,0),PS.Vec3d(.119,.001,0),PS.Vec3d(.119,0,.001)]
    x=PS.upload(BACKEND,PS.Position.(points))
    external=PS.upload(BACKEND,fill(PS.Vec3f(0,0,-1),3))
    base=PS.upload(BACKEND,fill(PS.Vec3f(0,0,-1),2))
    grip=PS.NodalAssemblyGrip(BACKEND,[1,3],points[[1,3]];
        weights=[.25,.75],stiffness=1e4,force_limit=2)
    one=PS.upload(BACKEND,[1f0])
    PS.launch!(BACKEND,PS.assembly_grip_targets!,
        (grip.targets,grip.reference,PS.Position(PS.Vec3d(-.01,0,0)),one),2)
    PS.sample_assembly_grip!(BACKEND,external,base,x,grip)
    force=Array(external)
    @test force≈[PS.Vec3f(-.5,0,-1),PS.Vec3f(0,0,-1),PS.Vec3f(-1.5,0,-1)]
    @test sum(norm(force[i]-PS.Vec3f(0,0,-1)) for i in (1,3))≈2f0
    # Trial positions can change without resampling the external actuator.
    copyto!(x,PS.Position.([p+PS.Vec3d(.001,0,0) for p in points]))
    @test Array(external)==force
    PS.sample_assembly_grip!(BACKEND,external,base,x,grip;engaged=false)
    @test Array(external)==fill(PS.Vec3f(0,0,-1),3)
    for (ids,ps,ws,k,limit) in (([1,1],points[[1,1]],[.5,.5],1.,1.),
            ([1],points[[1]],[2.],1.,1.),([0],points[[1]],[1.],1.,1.),
            ([1],points[[1]],[1.],0.,1.),([1],points[[1]],[1.],1.,Inf))
        @test_throws ArgumentError PS.NodalAssemblyGrip(BACKEND,ids,ps;weights=ws,stiffness=k,force_limit=limit)
    end
    @test_throws DimensionMismatch PS.sample_assembly_grip!(BACKEND,external,external,x,grip)
    far=PS.NodalAssemblyGrip(BACKEND,[4],points[[1]];stiffness=1.,force_limit=1.)
    @test_throws BoundsError PS.sample_assembly_grip!(BACKEND,external,PS.upload(BACKEND,[PS.Vec3f(0)]),x,far)
end

@testset "assembly actuator drives unconstrained material DOFs through coupled Newton" begin
    b=ModelBuilder(;gravity=PS.Vec3f(0))
    points,tets=tet_box((.002,.002,.002),(2,2,2);origin=PS.Vec3d(.1,0,0))
    add_deformable!(b,points,tets;young=1e5,fixed=p->p[1]<=.100001)
    world=WorldSolver(BACKEND,b;newton=6,iterations=32,cg_tolerance=1e-3,
        newton_tolerance=1e-8)
    state=WorldState(BACKEND,b)
    PS.prime!(world,state)
    nodes=findall(p->p[1]>=.101999,b.deformables[1].x)
    grip=PS.NodalAssemblyGrip(BACKEND,nodes,b.deformables[1].initial[nodes];
        stiffness=100.,force_limit=.01)
    base=PS.upload(BACKEND,fill(PS.Vec3f(0),length(nodes)))
    one=PS.upload(BACKEND,[1f0])
    PS.launch!(BACKEND,PS.assembly_grip_targets!,
        (grip.targets,grip.reference,PS.Position(PS.Vec3d(0,1e-4,0)),one),length(nodes))
    PS.sample_assembly_grip!(BACKEND,world.deformables.external,base,state.deformables.x,grip)
    before=Array(state.deformables.x)
    held=Array(world.deformables.external)
    stepper=Stepper(world,state,.001,1;reusable_newton=true,split_newton=true)
    PS.run!(stepper); PS.waitfor!(stepper)
    after=Array(state.deformables.x)
    @test all(p->all(isfinite,p.hi) && all(isfinite,p.lo),after)
    @test sum((after[i]-before[i])[2] for i in nodes)>0
    @test all(i->b.deformables[1].fixed[i]==0 || after[i]==before[i],eachindex(after))
    @test Array(world.deformables.external)==held
    PS.sample_assembly_grip!(BACKEND,world.deformables.external,base,state.deformables.x,grip;engaged=false)
    @test all(iszero,Array(world.deformables.external))
end
