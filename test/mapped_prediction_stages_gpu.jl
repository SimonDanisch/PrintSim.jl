using PrintSim,Test,LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

# Keep prediction asynchronous; reading inertia here is deliberately delayed
# until after surface rebuilding. This isolates the full-step ordering failure
# without involving contact forces, coarse preparation, or CG.
function mapped_prediction_stage_probe(backend,maps;prefix=true)
    b=ModelBuilder()
    axes=(PS.Vec3d(1,0,0),PS.Vec3d(0,1,0),PS.Vec3d(0,0,1))
    add_mesh!(b,box_mesh(PS.Vec3d(0,0,-.005),axes,(.1,.1,.005));static=.5)
    points,tets=tet_box((.01,.01,.003),(2,2,2))
    gap=resting_gap(1000*.003*9.81)
    add_deformable!(b,points,tets;density=1000,young=1e6,place=p->p+PS.Vec3d(0,0,gap))
    world=WorldSolver(backend,b;newton=2,iterations=8,cg_tolerance=1e-2,surface_maps=maps)
    state=WorldState(backend,b)
    PS.prime!(world,state)
    f,a=world.deformables,world.bodies
    if prefix in (true,:initialize)
        PS.launch!(backend,PS.initialize_newton_progress!,(world.energy,nothing),1)
    end
    if prefix in (true,:zero)
        PS.launch!(backend,PS.zero!,(world.energy.cg_iterations,),2)
    end
    if prefix===true
        PS.launch!(backend,PS.motor_clock!,(a.motor_state,a.motors,1f-3),a.motors.motors)
        PS.launch!(backend,PS.affine_predict!,(state.bodies,a.anchor,a.inertia,a.model,a.one,a.gravity,1f-3),4a.model.bodies)
    end
    PS.launch!(backend,PS.fem_predict!,(state.deformables,f.anchor,f.inertia,f.model,f.one,f.gravity,1f-3),f.model.nodes)
    PS.world_surface!(backend,world,state)
    return (;x=Array(state.deformables.x),inertia=Array(f.inertia),gravity=f.gravity)
end

@testset "asynchronous mapped prediction before contact" begin
    for prefix in (false,true),maps in (nothing,Dict{Int,Any}())
        println("PREDICTION_STAGE_BEGIN prefix=",prefix," mapped=",maps!==nothing); flush(stdout)
        result=mapped_prediction_stage_probe(BACKEND,maps;prefix)
        precise(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
        expected=PS.Vec3d(1f-3^2*result.gravity)
        @test maximum(norm.(precise.(result.inertia).-precise.(result.x).-Ref(expected)))<1e-12
        println("PREDICTION_STAGE_END prefix=",prefix," mapped=",maps!==nothing); flush(stdout)
    end
end
