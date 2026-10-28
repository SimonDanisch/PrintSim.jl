using PrintSim,Test,LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "device local solves agree with host block factors" begin
    b=ModelBuilder()
    points,tets=tet_box((0.008,0.006,0.002),(3,2,2))
    add_deformable!(b,points,tets;young=1e5,fixed=p->p[1]==0)
    solver=FEMSolver(BACKEND,b;schwarz=true)
    state=FEMState(BACKEND,b)
    PS.fem_linearize!(BACKEND,solver,state)
    copyto!(solver.inertia,state.x)
    PS.fem_rhs!(BACKEND,solver,state,1f-3)
    PS.fem_prepare_preconditioner!(BACKEND,solver)
    reference=PS.FEMSchwarz(Array,b)
    model=PS.Adapt.adapt(Array,solver.model)
    inverse,gradients=Array(solver.inverse),Array(solver.gradients)
    for i in 1:model.nodes
        PS.schwarz_assemble_node!(reference,inverse,gradients,model,Int32(i))
    end
    for p in 1:reference.patches
        PS.schwarz_factor_patch!(reference,Int32(p))
    end
    @test Array(solver.schwarz.fallback)==reference.fallback
    @test Array(solver.schwarz.factor) ≈ reference.factor rtol=3e-4
    residual=[PS.Vec3f(sin(i),cos(i),sin(i/3)) for i in 1:model.nodes]
    copyto!(solver.r,residual)
    PS.launch!(BACKEND,PS.fem_precondition!,
        (solver.z,solver.r,solver.inverse,solver.model),model.nodes)
    PS.launch!(BACKEND,PS.schwarz_apply!,
        (solver.z,solver.r,solver.schwarz,solver.model),solver.schwarz.patches)
    expected=[model.fixed[i]==0x00 ? inverse[i]*residual[i] : zero(PS.Vec3f) for i in 1:model.nodes]
    for p in 1:reference.patches
        PS.schwarz_apply_patch!(expected,residual,reference,model,Int32(p))
    end
    @test Array(solver.z) ≈ expected rtol=5e-4
    @test all(iszero,Array(solver.z)[findall(!iszero,model.fixed)])
end
