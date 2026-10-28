using PrintSim,Test
import PrintSim as PS
import Mantle
using GeometryBasics: Vec3
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "swept broad phase includes non-neighbor sheets and thickness" begin
    b=ModelBuilder()
    grid=SheetGrid(0.01,0.01,2,2)
    for (x,z) in ((0.0,0.0),(0.1,0.0),(0.0,0.001))
        add_sheet!(b,grid,SheetMaterial();place=p->Vec3(p[1]+x,p[2],z))
    end
    ccd=PS.SweptCollision(BACKEND,b)
    finish=Array(ccd.finish)
    for i in vertices(b.sheets[3])
        finish[i]=Position(PS.Vec3d(b.x[i])-PS.Vec3d(0,0,0.002))
    end
    copyto!(ccd.finish,finish)
    function declare_sweep(target;batch=nothing)
        PS.launch!(target,PS.zero!,(ccd.status,),1)
        PS.launch!(target,PS.reset_step_bound!,(ccd.bound,),1)
        for level in 0:ccd.depth
            PS.launch!(target,PS.swept_refit!,(ccd,Int32(level)),length(ccd.left))
        end
        if batch===nothing
            PS.launch!(target,PS.swept_collisions!,(ccd,),ccd.query_count)
        else
            PS.launch_indexed_batches!(target,PS.swept_collisions!,(ccd,false),ccd.query_count;batch)
        end
    end
    declare_sweep(BACKEND)
    bound=reinterpret(Float32,only(Array(ccd.bound)))
    status=only(Array(ccd.status))
    @test 0 < bound <= 0.45f0
    @test status & UInt32(2)==0
    graph=Mantle.Graph(Mantle.todevice(BACKEND))
    declare_sweep(graph;batch=3)
    plan=Mantle.Plan(graph)
    Mantle.record!(plan);Mantle.run!(plan);Mantle.waitfor!(plan)
    @test only(Array(ccd.bound))==reinterpret(UInt32,bound)
    @test only(Array(ccd.status))==status
    # A stationary intersection must be diagnosed at initialization, even
    # though common translation is safely skipped in subsequent sweeps.
    positions=Array(ccd.start)
    for (a,c) in zip(vertices(b.sheets[1]),vertices(b.sheets[3]))
        positions[c]=positions[a]
    end
    copyto!(ccd.start,positions);copyto!(ccd.finish,positions)
    PS.launch!(BACKEND,PS.zero!,(ccd.status,),1)
    for level in 0:ccd.depth
        PS.launch!(BACKEND,PS.swept_refit!,(ccd,Int32(level)),length(ccd.left))
    end
    PS.launch!(BACKEND,PS.swept_collisions!,(ccd,true),ccd.query_count)
    @test only(Array(ccd.status)) & UInt32(2)!=0
    report=PS.ccd_failure_report(ccd;limit=1)
    @test length(report)==1 && report[1].flags & UInt32(2)!=0
    @test Set(report[1].owners)==Set((Int32(1),Int32(3)))
    # Initialization completes each small batch before reporting progress and
    # stops at a real zero-time blocker, retaining its diagnostic state.
    PS.launch!(BACKEND,PS.zero!,(ccd.status,),1)
    PS.launch!(BACKEND,PS.zero!,(ccd.blocking,),ccd.query_count)
    PS.launch!(BACKEND,PS.zero!,(ccd.blocking_reason,),ccd.query_count)
    progress=NamedTuple[]
    processed=PS.initial_ccd_batches!(BACKEND,ccd;batch=3,on_batch=p->push!(progress,p))
    @test 0<processed<=ccd.query_count
    @test last(progress).last==processed && last(progress).status & UInt32(2)!=0
    @test all(p.last-p.first+1<=3 for p in progress)
    @test all(progress[k].first==progress[k-1].last+1 for k in 2:length(progress))
    @test_throws ArgumentError PS.initial_ccd_batches!(BACKEND,ccd;batch=0)
end
