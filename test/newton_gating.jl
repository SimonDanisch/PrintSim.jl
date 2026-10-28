using PrintSim,Test
import PrintSim as PS
import Mantle
using GeometryBasics: Vec3f
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "device Newton gating preserves accepted surface and replay state" begin
    b=ModelBuilder(;gravity=Vec3f(0))
    add_sheet!(b,SheetGrid(0.01,0.01,3,3),SheetMaterial())
    points,tets=tet_box((0.002,0.002,0.002),(2,2,2);origin=PS.Vec3d(0.1,0,0))
    add_deformable!(b,points,tets;young=1e5,fixed=p->true)
    axes=(PS.Vec3d(1,0,0),PS.Vec3d(0,1,0),PS.Vec3d(0,0,1))
    add_body!(TEST_DEVICE, b,box_mesh(PS.Vec3d(0.2,0,0),axes,(0.001,0.001,0.001));fixed=true)
    solver=WorldSolver(BACKEND,b;newton=4,iterations=8,newton_tolerance=1e-8,elastic_schwarz=true,
        newton_motion_limit=15e-6)
    state=WorldState(BACKEND,b)
    PS.prime!(solver,state)
    history=PS.upload(BACKEND,zeros(Float32,4,4))
    for reusable_newton in (false,true)
      stepper=Stepper(solver,state,1e-3,1;solve_history=history,reusable_newton)
      for _ in 1:2
        PS.run!(stepper);PS.waitfor!(stepper)
        status=Array(solver.energy.newton_status)
        @test status==UInt32[0,1]
        progress=Array(solver.energy.newton_progress)
        @test all(iszero,progress[:,2:end])
        @test Array(history)[:,2:end]==repeat(Float32[0,0,0,1],1,3)
      end
    end
end
