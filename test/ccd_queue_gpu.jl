using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")
ccd_queue_test_policy()=isdefined(@__MODULE__,:CCD_TEST_POLICY) ? CCD_TEST_POLICY : PS.Mantle.Overlap()
ccd_queue_test_compact()=isdefined(@__MODULE__,:CCD_TEST_COMPACT_QUERIES) ? CCD_TEST_COMPACT_QUERIES : false

@testset "compact query tiles keep declaration size independent of tile count" begin
    b=ModelBuilder()
    for z in (0.,.01)
        add_sheet!(b,SheetGrid(.01,.01,8,8),SheetMaterial();place=p->PS.Vec3d(p[1],p[2],z))
    end
    ccd=PS.SweptCollision(BACKEND,b)
    queue=PS.RuntimeCCDQueue(BACKEND,ccd)
    function graph_size(batch,compact)
        g=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
        PS.queued_swept_collisions!(g,queue,ccd;query_batch=batch,compact_queries=compact)
        length(g.passes)
    end
    @test graph_size(1,true)==graph_size(17,true)
    # Mantle elides loop bookkeeping for a one-iteration body.
    @test graph_size(2048,true)<=graph_size(17,true)
    @test graph_size(1,true)<graph_size(1,false)÷8
end

@testset "moving pair queue matches fused CCD and refreshes counts on replay" begin
    b=ModelBuilder()
    for z in (0.,.01)
        add_sheet!(b,SheetGrid(.01,.01,2,2),SheetMaterial();place=p->PS.Vec3d(p[1],p[2],z))
    end
    initial=PS.SweptCollision(BACKEND,b)
    host=PS.ccd_snapshot(initial)
    queued=PS.restore_ccd_snapshot(BACKEND,host)
    work=PS.RuntimeCCDQueue(BACKEND,queued;capacity=64,pair_batch=3)
    function refit!(ccd)
        for level in 0:ccd.depth
            PS.launch!(BACKEND,PS.swept_refit!,(ccd,Int32(level)),length(ccd.left))
        end
    end
    graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
    PS.queued_swept_collisions!(graph,work,queued;query_batch=3,compact_queries=ccd_queue_test_compact())
    plan=PS.Mantle.Plan(graph;policy=ccd_queue_test_policy());PS.Mantle.record!(plan)
    totals=Int[]
    for lift in (.0,.02,.0)
        finish=copy(host.data.finish)
        for i in 1:4
            p=finish[i]
            finish[i]=PS.Position(PS.Vec3d(p.hi)+PS.Vec3d(p.lo)+PS.Vec3d(0,0,lift))
        end
        reference=PS.restore_ccd_snapshot(BACKEND,host)
        for ccd in (queued,reference)
            copyto!(ccd.finish,finish)
            copyto!(ccd.bound,UInt32[reinterpret(UInt32,1f0)])
            copyto!(ccd.status,UInt32[0])
            copyto!(ccd.blocking,zeros(Int32,ccd.query_count))
            copyto!(ccd.blocking_reason,zeros(UInt32,ccd.query_count))
            refit!(ccd)
        end
        PS.launch!(BACKEND,PS.swept_collisions!,(reference,false),reference.query_count)
        PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
        @test only(Array(work.overflow))==0
        @test only(Array(work.work.invalid))==0
        @test only(Array(queued.status))==only(Array(reference.status))
        @test only(Array(queued.bound))==only(Array(reference.bound))
        push!(totals,only(Array(work.total)))
    end
    @test totals[1]==totals[3]<totals[2]
    # Capacity overflow cannot turn a colliding trajectory into a safe step.
    too_small=PS.RuntimeCCDQueue(BACKEND,queued;capacity=1,pair_batch=1)
    copyto!(queued.finish,host.data.start)
    copyto!(queued.status,UInt32[0]);copyto!(queued.bound,UInt32[reinterpret(UInt32,1f0)])
    refit!(queued)
    PS.queued_swept_collisions!(BACKEND,too_small,queued;query_batch=2)
    @test only(Array(too_small.overflow))>0
    @test only(Array(queued.status)) & UInt32(2)!=0
    @test only(Array(queued.status)) & UInt32(4)!=0
    @test only(Array(queued.bound))==0
    @test only(Array(too_small.chunks))==0
    # The SAME recorded graph must recover from an overflow after inputs
    # return within capacity; neither a stale count nor stale flags may win.
    bounded=PS.RuntimeCCDQueue(BACKEND,queued;capacity=4,pair_batch=3)
    bounded_graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
    PS.queued_swept_collisions!(bounded_graph,bounded,queued;query_batch=3,compact_queries=ccd_queue_test_compact())
    bounded_plan=PS.Mantle.Plan(bounded_graph;policy=ccd_queue_test_policy());PS.Mantle.record!(bounded_plan)
    for (lift,overflow) in ((0.,false),(.02,true),(0.,false))
        finish=copy(host.data.finish)
        for i in 1:4
            p=finish[i]
            finish[i]=PS.Position(PS.Vec3d(p.hi)+PS.Vec3d(p.lo)+PS.Vec3d(0,0,lift))
        end
        copyto!(queued.finish,finish)
        copyto!(queued.status,UInt32[0]);copyto!(queued.bound,UInt32[reinterpret(UInt32,1f0)])
        refit!(queued)
        PS.Mantle.run!(bounded_plan);PS.Mantle.waitfor!(bounded_plan)
        @test (only(Array(bounded.overflow))>0)==overflow
        @test (only(Array(queued.status)) & UInt32(2)!=0)==overflow
        @test (only(Array(queued.status)) & UInt32(4)!=0)==overflow
        @test only(Array(bounded.total))==(overflow ? 0 : totals[1])
    end
    active=PS.upload(BACKEND,UInt32[1])
    conditional_graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
    PS.queued_swept_collisions!(PS.ConditionalTarget(conditional_graph,active),work,queued;query_batch=3,compact_queries=ccd_queue_test_compact())
    conditional_plan=PS.Mantle.Plan(conditional_graph;policy=ccd_queue_test_policy());PS.Mantle.record!(conditional_plan)
    for enabled in UInt32[1,0,1]
        copyto!(active,[enabled])
        PS.Mantle.run!(conditional_plan);PS.Mantle.waitfor!(conditional_plan)
        @test only(Array(work.chunks))==(enabled==0 ? 0 : cld(totals[1],work.pair_batch))
    end
end

@testset "tiled pair reports retain global query identities" begin
    b=ModelBuilder()
    for z in (0.,.01)
        add_sheet!(b,SheetGrid(.01,.01,2,2),SheetMaterial();place=p->PS.Vec3d(p[1],p[2],z))
    end
    ccd=PS.SweptCollision(BACKEND,b)
    @test ccd.query_count==4
    queue=PS.RuntimeCCDQueue(BACKEND,ccd;capacity=4,pair_batch=1)
    copyto!(queue.work.count,Int32[1,0,1,1])
    copyto!(queue.work.offset,Int64[0,1,1,2])
    copyto!(queue.work.pairs,[(Int32(1),Int32(2)),(Int32(3),Int32(4)),
        (Int32(4),Int32(1)),(Int32(0),Int32(0))])
    copyto!(queue.work.flags,UInt32[2,1,2,0])
    copyto!(queue.total,Int32[3])
    for compact_queries in (false,true),batch in (1,3)
        copyto!(ccd.blocking,zeros(Int32,4));copyto!(ccd.blocking_reason,zeros(UInt32,4))
        target=compact_queries ? PS.Mantle.Graph(PS.Mantle.todevice(BACKEND)) : BACKEND
        PS.report_runtime_ccd_queue!(target,queue,ccd;query_batch=batch,compact_queries)
        if compact_queries
            plan=PS.Mantle.Plan(target;policy=ccd_queue_test_policy())
            PS.Mantle.record!(plan);PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
        end
        @test Array(ccd.blocking)==Int32[2,0,4,1]
        @test Array(ccd.blocking_reason)==UInt32[2,0,1,2]
    end
end

@testset "moving CCD cannot discard sub-ulp inclined-face motion" begin
    b=ModelBuilder()
    for z in (0.,.1)
        add_sheet!(b,SheetGrid(.01,.01,2,2),SheetMaterial();place=p->PS.Vec3d(p[1],p[2],z))
    end
    ccd=PS.SweptCollision(BACKEND,b)
    gap=1e-10
    coordinates=PS.Vec3d[(0,0,0),(0,0,-.011),(0,-.011,0),(1,1,1),
        (.01,-.02,.01+gap),(.02,.01,-.03+gap),(-.03,.01,.02+gap),(2,2,2)]
    initial=PS.Position.(coordinates)
    finish=copy(initial)
    for i in 5:7
        finish[i]=PS.Position(coordinates[i]-PS.Vec3d(0,0,2gap))
    end
    faces=Array(ccd.faces);faces[1]=(1,2,3);faces[2]=(5,6,7)
    owners=Array(ccd.owner);owners[1]=1;owners[2]=2
    copyto!(ccd.start,initial);copyto!(ccd.finish,finish)
    copyto!(ccd.faces,faces);copyto!(ccd.owner,owners)
    copyto!(ccd.radius,zeros(Float32,length(ccd.radius)))
    work=(;pairs=PS.upload(BACKEND,[(Int32(1),Int32(2))]),flags=PS.upload(BACKEND,UInt32[0]))
    one=PS.upload(BACKEND,Int32[1])
    PS.launch!(BACKEND,PS.ccd_queue_checks!,(work,ccd,one,one,Int32(1),true),1)
    bound=reinterpret(Float32,only(Array(ccd.bound)))
    # The face moves through the stationary origin at t=1/2; radius is zero.
    @test 0<bound<.5
    @test only(Array(ccd.status)) & UInt32(2)==0
end
