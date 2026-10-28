using PrintSim,Test,LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "mapped one-hot coupled contact agrees with legacy FE path" begin
    b=ModelBuilder()
    axes=(PS.Vec3d(1,0,0),PS.Vec3d(0,1,0),PS.Vec3d(0,0,1))
    add_mesh!(b,box_mesh(PS.Vec3d(0,0,-.005),axes,(.1,.1,.005));static=.5)
    points,tets=tet_box((.01,.01,.003),(2,2,2))
    gap=resting_gap(1000*.003*9.81)
    add_deformable!(b,points,tets;density=1000,young=1e6,place=p->p+PS.Vec3d(0,0,gap))
    positions=Vector{PS.Position}[]
    forces=Float64[]
    diagnostics=NamedTuple[]
    for maps in (nothing,Dict{Int,Any}())
        solver=WorldSolver(BACKEND,b;newton=2,iterations=8,cg_tolerance=1e-2,surface_maps=maps)
        state=WorldState(BACKEND,b)
        PS.prime!(solver,state)
        history=PS.upload(BACKEND,zeros(Float32,4,solver.newton))
        linearized=NamedTuple[]
        initial_cg=NamedTuple[]
        predicted=NamedTuple[]
        on_predicted=(w,st,dt)->push!(predicted,(;dt,gravity=w.deformables.gravity,
            one=Array(w.deformables.one),fixed=Array(w.deformables.model.fixed),
            x=Array(st.deformables.x),anchor=Array(w.deformables.anchor),
            inertia=Array(w.deformables.inertia)))
        on_linearized=(iteration,w,st)->push!(linearized,(;iteration,
            x=Array(st.deformables.x),anchor=Array(w.deformables.anchor),inertia=Array(w.deformables.inertia),
            rhs=Array(w.deformables.r),inverse=Array(w.deformables.inverse),
            contact_force=Array(w.bodies.contacts.force),contact_block=Array(w.bodies.contacts.block)))
        on_cg_initial=(iteration,w,st)->push!(initial_cg,(;iteration,
            z=Array(w.deformables.z),rhs=Array(w.deformables.r),scalars=Array(w.energy.cg_scalars),
            coarse=Array(w.deformables.coarse.scratch)))
        traced=get(ENV,"PRINTSIM_PARITY_TRACE","false")=="true"
        prediction_mode=get(ENV,"PRINTSIM_PARITY_PREDICTION_MODE","readback")
        prediction_mode in ("readback","waitidle","host","none") || error("unknown prediction trace mode")
        prediction_callback=prediction_mode=="readback" ? on_predicted :
            prediction_mode=="waitidle" ? ((w,st,dt)->PS.Mantle.waitidle(TEST_DEVICE)) :
            prediction_mode=="host" ? ((w,st,dt)->nothing) : nothing
        PS.step!(BACKEND,solver,state,1e-3;solve_history=history,
            on_linearized=traced ? on_linearized : nothing,on_cg_initial=traced ? on_cg_initial : nothing,
            on_predicted=traced ? prediction_callback : nothing)
        push!(positions,Array(state.deformables.x))
        push!(forces,sum(Array(solver.bodies.contacts.lagged)))
        push!(diagnostics,(linear=Array(history),newton=Array(solver.energy.newton_progress),
            energy=Array(solver.energy.sum),scalars=Array(solver.energy.cg_scalars),
            residual=Array(solver.deformables.r),coarse=Array(solver.deformables.coarse.scratch),linearized,initial_cg,predicted,traced,prediction_mode))
        @test all(p->all(isfinite,p.hi)&&all(isfinite,p.lo),positions[end])
        @test all(iszero,Array(solver.fem_contacts.overflow))
    end
    directory=get(ENV,"PRINTSIM_DIAGNOSTIC_DIR",nothing)
    if directory!==nothing
        mkpath(directory)
        path=joinpath(directory,"mapped_fem_full_step.jls")
        isfile(path) && error("preserving previous step diagnostics; choose a new directory")
        PS.Serialization.serialize(path,(;positions,forces,diagnostics,scope=:two_newton_eight_cg_diagnostic))
    end
    precise(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
    # Validate prediction independently of contact/CG convergence. Keep this
    # after serialization so a failed prediction retains its complete evidence.
    for diagnostic in diagnostics, prediction in diagnostic.predicted
        acceleration_step=PS.Vec3d(prediction.dt^2*prediction.gravity)
        @test prediction.one==Float32[1]
        @test all(eachindex(prediction.x)) do i
            expected=prediction.fixed[i]==0x00 ? acceleration_step : zero(PS.Vec3d)
            norm((precise(prediction.inertia[i])-precise(prediction.x[i]))-expected)<1e-12
        end
    end
    @test maximum(norm.(precise.(positions[1]).-precise.(positions[2])))<3e-10
    @test forces[1]>0
    @test forces[2]≈forces[1] rtol=1e-5
end
