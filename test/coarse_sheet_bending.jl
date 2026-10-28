using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select an explicit software or coordinated GPU backend")

@testset "unresolved cylinder calibration uses finite discrete-shell hinges" begin
    material=SheetMaterial()
    for (nu,nv) in ((2,2),(2,5),(5,2))
        grid=SheetGrid(.01,.02,nu,nv)
        b=ModelBuilder(;gravity=PS.Vec3f(0))
        add_sheet!(b,grid,material)
        points=[PS.Vec3d(PS.flat(grid,i,j)[1],PS.flat(grid,i,j)[2],0)
            for j in 1:nv for i in 1:nu][:]
        hinges=PS.grid_hinges(PS.grid_triangles(grid))
        @test PS.bending_calibration(grid,hinges)==(3.,3.)
        @test all(isfinite,b.hinge_stiffness)
        @test all(>(0),b.hinge_stiffness)
        @test b.hinge_stiffness≈[3*PS.bending_stiffness(material)*PS.hinge_weight(points,h...) for h in hinges]
        solver=WorldSolver(BACKEND,b;newton=2,iterations=8,cg_tolerance=.01,newton_tolerance=1e-8)
        state=WorldState(BACKEND,b);PS.prime!(solver,state)
        history=PS.upload(BACKEND,zeros(Float32,4,2))
        stepper=Stepper(solver,state,.001,1;reusable_newton=true,solve_history=history)
        PS.run!(stepper);PS.waitfor!(stepper)
        @test all(p->all(isfinite,p.hi)&&all(isfinite,p.lo),Array(state.sheets.x))
    end
    b=ModelBuilder()
    @test_throws ArgumentError add_sheet!(b,SheetGrid(.01,.01,1,3),material)
    @test_throws ArgumentError add_sheet!(b,SheetGrid(0.,.01,3,3),material)
    @test isempty(b.x) && isempty(b.sheets)
end
