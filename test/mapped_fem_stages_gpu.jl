using PrintSim,Test,LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

function mapped_fem_stage_probe(backend,maps)
    b=ModelBuilder()
    axes=(PS.Vec3d(1,0,0),PS.Vec3d(0,1,0),PS.Vec3d(0,0,1))
    add_mesh!(b,box_mesh(PS.Vec3d(0,0,-.005),axes,(.1,.1,.005));static=.5)
    points,tets=tet_box((.01,.01,.003),(2,2,2))
    gap=resting_gap(1000*.003*9.81)
    add_deformable!(b,points,tets;density=1000,young=1e6,place=p->p+PS.Vec3d(0,0,gap))
    world=WorldSolver(backend,b;newton=2,iterations=8,cg_tolerance=1e-2,surface_maps=maps)
    state=WorldState(backend,b)
    PS.prime!(world,state)
    f,a,s=world.deformables,world.bodies,world.sheets
    nf=f.model.nodes
    npoints=length(a.colliders.point_owner)
    PS.launch!(backend,PS.fem_predict!,(state.deformables,f.anchor,f.inertia,f.model,f.one,f.gravity,1f-3),nf)
    PS.world_surface!(backend,world,state)
    PS.launch_indexed_batches!(backend,PS.find_body_contacts!,
        (a.contacts,a.colliders,a.model,state.bodies,a.anchor,a.one,a.roughness,a.inertia),npoints)
    PS.index_body_contacts!(backend,a.contacts,a.colliders,a.model)
    index=PS.fem_contact_index(world.fem_contacts)
    model=PS.fem_contact_model(f.model,world.fem_contacts)
    PS.launch!(backend,PS.fem_contact_endpoints!,
        (index,model,a.colliders,a.contacts,s.touches,s.faces,Int32(0)),index.entries)
    PS.compact!(backend,index)
    motion=PS.DeformingMeshes(PS.MovingMeshes(a.model,state.bodies,a.anchor,world.coupling),f.model,state.deformables,f.anchor)
    PS.launch!(backend,PS.body_contact_forces!,
        (a.contacts,a.colliders,a.model,state.bodies,a.anchor,a.one,a.roughness,a.tangential,false,motion),npoints)
    PS.fem_linearize!(backend,f,state.deformables)
    PS.fem_rhs!(backend,f,state.deformables,1f-3)
    before=(rhs=Array(f.r),inverse=Array(f.inverse))
    PS.world_fem_contact_rhs!(backend,world)
    after=(rhs=Array(f.r),inverse=Array(f.inverse))
    direction=PS.Vec3f[PS.Vec3f(sin(i),cos(i),sin(2i)) for i in 1:nf]
    copyto!(f.p,PS.upload(backend,direction))
    PS.fem_product!(backend,f,1f-3)
    base_product=Array(f.q)
    PS.world_fem_contact_product!(backend,world)
    product=Array(f.q)
    copyto!(f.q,PS.upload(backend,direction.*1f-6))
    PS.launch!(backend,PS.reset_step_bound!,(world.step_bound,),1)
    PS.world_fem_contact_bound!(backend,world)
    bound=Array(world.step_bound)
    return (;before,after,base_product,product,bound,points=Array(a.colliders.points),
        contact_mesh=Array(a.contacts.mesh),contact_triangle=Array(a.contacts.triangle),
        contact_force=Array(a.contacts.force),contact_block=Array(a.contacts.block),
        keys=Array(index.keys),count=Array(index.count))
end

@testset "mapped FE contact stage parity" begin
    legacy=mapped_fem_stage_probe(BACKEND,nothing)
    mapped=mapped_fem_stage_probe(BACKEND,Dict{Int,Any}())
    directory=get(ENV,"PRINTSIM_DIAGNOSTIC_DIR",nothing)
    if directory!==nothing
        mkpath(directory)
        path=joinpath(directory,"mapped_fem_stages.jls")
        isfile(path) && error("preserving previous stage diagnostics; choose a new directory")
        PS.Serialization.serialize(path,(;legacy,mapped,scope=:toy_stage_parity_only))
    end
    for name in (:points,:contact_mesh,:contact_triangle,:contact_force,:contact_block)
        @test getproperty(mapped,name)==getproperty(legacy,name)
    end
    @test mapped.before.rhs≈legacy.before.rhs rtol=1e-6
    @test mapped.before.inverse≈legacy.before.inverse rtol=1e-6
    @test mapped.after.rhs≈legacy.after.rhs rtol=1e-6
    @test mapped.after.inverse≈legacy.after.inverse rtol=1e-6
    @test mapped.base_product≈legacy.base_product rtol=1e-6
    @test mapped.product≈legacy.product rtol=1e-6
    @test mapped.bound==legacy.bound
end
