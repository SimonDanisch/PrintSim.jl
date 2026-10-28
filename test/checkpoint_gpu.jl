using PrintSim,Test,LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "cold coupled restart retains state, friction and motor control" begin
    b=ModelBuilder()
    axes=(PS.Vec3d(1,0,0),PS.Vec3d(0,1,0),PS.Vec3d(0,0,1))
    add_mesh!(b,box_mesh(PS.Vec3d(0,0,-.005),axes,(.1,.1,.005));static=.5)
    body=add_body!(TEST_DEVICE, b,box_mesh(PS.Vec3d(.04,0,.03),axes,(.005,.005,.005));mass=.01)
    add_joint!(b,body;axis=PS.Vec3d(1,0,0))
    add_stepper!(b,body;axis=PS.Vec3d(1,0,0),torque=1e-3)
    points,tets=tet_box((.01,.01,.003),(2,2,2))
    gap=resting_gap(1000*.003*9.81)
    add_deformable!(b,points,tets;density=1000,young=1e6,place=p->p+PS.Vec3d(0,0,gap))
    add_sheet!(b,SheetGrid(.01,.01,3,3),SheetMaterial();place=p->p+PS.Vec3d(-.03,0,.03))
    build()=WorldSolver(BACKEND,b;newton=2,iterations=8,cg_tolerance=1e-2,surface_maps=Dict{Int,Any}())
    control()=MotorSequence(BACKEND,reshape(Int32[5],1,1);rpm=3.,timestep=1e-3)
    world,state,sequence=build(),WorldState(BACKEND,b),control()
    PS.prime!(world,state)
    PS.launch!(BACKEND,PS.motor_sequence!,(sequence,world.bodies.motor_state),1)
    PS.step!(world,state,1e-3)
    digest=repeat("a",64)
    checkpoint=PS.state_checkpoint(world,state;time=.001,step=1,model_digest=digest,engine_digest=digest,control=sequence)
    cold,cold_state,cold_sequence=build(),WorldState(BACKEND,b),control()
    PS.prime!(cold,cold_state)
    @test_throws ArgumentError PS.restore_state_checkpoint!(cold,cold_state,checkpoint;
        model_digest=repeat("b",64),engine_digest=digest,control=cold_sequence)
    restored=PS.restore_state_checkpoint!(cold,cold_state,checkpoint;
        model_digest=digest,engine_digest=digest,control=cold_sequence)
    @test restored==(;time=.001,step=1)
    bindings=PS.checkpoint_array_bindings(PS.checkpoint_roots(cold,cold_state,cold_sequence))
    @test all(path->Array(bindings[path])==checkpoint.arrays[path],keys(bindings))
    mktempdir() do directory
        path=PS.save_state_checkpoint(joinpath(directory,"restart.jls"),world,state;
            time=.001,step=1,model_digest=digest,engine_digest=digest,control=sequence)
        @test isfile(path)
        @test_throws ArgumentError PS.save_state_checkpoint(path,world,state;
            time=.001,step=1,model_digest=digest,engine_digest=digest,control=sequence)
    end
    for (w,s,c) in ((world,state,sequence),(cold,cold_state,cold_sequence))
        PS.launch!(BACKEND,PS.motor_sequence!,(c,w.bodies.motor_state),1)
        PS.step!(w,s,1e-3)
    end
    precise(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
    for (left,right) in ((state.sheets.x,cold_state.sheets.x),(state.sheets.v,cold_state.sheets.v),
            (state.bodies.q,cold_state.bodies.q),(state.bodies.v,cold_state.bodies.v),
            (state.deformables.x,cold_state.deformables.x),(state.deformables.v,cold_state.deformables.v))
        @test maximum(norm.(precise.(Array(left)).-precise.(Array(right))))<1e-10
    end
    @test Array(sequence.status)==Array(cold_sequence.status)
    @test Array(world.bodies.motor_state.commanded)==Array(cold.bodies.motor_state.commanded)
end
