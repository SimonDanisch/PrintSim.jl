using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select an explicit software or coordinated GPU backend")

@testset "bounded CCD stream matches complete pair queue, including replay" begin
    b=ModelBuilder()
    for z in (0.,.01)
        add_sheet!(b,SheetGrid(.01,.01,2,2),SheetMaterial();place=p->PS.Vec3d(p[1],p[2],z))
    end
    snapshot=PS.ccd_snapshot(PS.SweptCollision(BACKEND,b))
    for parallel in (false,true),(tile,capacity) in ((1,64),(3,64),(4,4))
        ccd=PS.restore_ccd_snapshot(BACKEND,snapshot)
        stream=PS.StreamingCCDQueue(BACKEND,ccd;query_batch=tile,capacity,pair_batch=1,primitive_parallel=parallel)
        @test stream.primitive_parallel==parallel
        @test length(stream.queue.work.count)==min(tile,ccd.query_count)
        @test length(stream.queue.work.pairs)==capacity
        active=PS.upload(BACKEND,UInt32[1])
        graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
        PS.queued_swept_collisions!(PS.ConditionalTarget(graph,active),stream,ccd)
        plan=PS.Mantle.Plan(graph;policy=PS.Mantle.InOrder());PS.Mantle.record!(plan)
        for lift in (0.,.02,0.)
            finish=copy(snapshot.data.finish)
            for i in 1:4
                p=finish[i]
                finish[i]=PS.Position(PS.Vec3d(p.hi)+PS.Vec3d(p.lo)+PS.Vec3d(0,0,lift))
            end
            reference=PS.restore_ccd_snapshot(BACKEND,snapshot)
            for scene in (ccd,reference)
                copyto!(scene.finish,finish)
                copyto!(scene.status,UInt32[0]);copyto!(scene.bound,UInt32[reinterpret(UInt32,1f0)])
                copyto!(scene.blocking,zeros(Int32,scene.query_count))
                copyto!(scene.blocking_reason,zeros(UInt32,scene.query_count))
                for level in 0:scene.depth
                    PS.launch!(BACKEND,PS.swept_refit!,(scene,Int32(level)),length(scene.left))
                end
            end
            complete=PS.RuntimeCCDQueue(BACKEND,reference;capacity=64,pair_batch=1)
            PS.queued_swept_collisions!(BACKEND,complete,reference;query_batch=3)
            PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
            @test Array(ccd.status)==Array(reference.status)
            @test Array(ccd.bound)==Array(reference.bound)
            @test Array(ccd.blocking)==Array(reference.blocking)
            @test Array(ccd.blocking_reason)==Array(reference.blocking_reason)
            @test only(Array(stream.total))==only(Array(complete.total))
            @test only(Array(stream.overflow))==0
            @test only(Array(stream.state.running))==0
            @test only(Array(stream.state.first))==ccd.query_count+1
            if tile==3 && lift==.02
                segmented=PS.restore_ccd_snapshot(BACKEND,PS.ccd_snapshot(ccd))
                copyto!(segmented.status,UInt32[0])
                copyto!(segmented.bound,UInt32[reinterpret(UInt32,1f0)])
                copyto!(segmented.blocking,zeros(Int32,segmented.query_count))
                copyto!(segmented.blocking_reason,zeros(UInt32,segmented.query_count))
                chunks=PS.StreamingCCDQueue(BACKEND,segmented;query_batch=1,capacity=64,pair_batch=1)
                PS.initialize_ccd_stream!(BACKEND,chunks,segmented)
                chunk_graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
                PS.advance_ccd_stream!(chunk_graph,chunks,segmented;iterations=1)
                chunk_plan=PS.Mantle.Plan(chunk_graph;policy=PS.Mantle.InOrder())
                PS.Mantle.record!(chunk_plan)
                for attempt in 1:PS.ccd_stream_iteration_limit(chunks,segmented)
                    PS.Mantle.run!(chunk_plan);PS.Mantle.waitfor!(chunk_plan)
                    @test only(Array(segmented.status)) & UInt32(8)==0
                    only(Array(chunks.state.running))==0 && break
                end
                @test only(Array(chunks.state.running))==0
                @test Array(segmented.bound)==Array(ccd.bound)
                @test Array(segmented.status)==Array(ccd.status)
                @test Array(segmented.blocking)==Array(ccd.blocking)
                @test Array(segmented.blocking_reason)==Array(ccd.blocking_reason)
                @test Array(chunks.total)==Array(stream.total)
                @test_throws ArgumentError PS.advance_ccd_stream!(BACKEND,chunks,segmented;iterations=0)
                @test_throws ArgumentError PS.advance_ccd_stream!(BACKEND,chunks,segmented;
                    iterations=PS.ccd_stream_iteration_limit(chunks,segmented)+1)
            end
            if capacity==64
                @test only(Array(stream.state.tile))==cld(ccd.query_count,tile)+1
            elseif lift>0
                @test only(Array(stream.state.retries))>0
            end
        end
        # Disabled Newton stages must clear the loop gate without touching
        # active-phase counters, even if stale state was deliberately poisoned.
        copyto!(active,UInt32[0]);copyto!(stream.state.running,Int32[1])
        copyto!(stream.total,Int64[-999])
        PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
        @test only(Array(stream.state.running))==0
        @test only(Array(stream.total))==-999
    end
    # An exhausted device-work budget is never a successfully tested prefix.
    scene=PS.restore_ccd_snapshot(BACKEND,snapshot)
    incomplete=PS.StreamingCCDQueue(BACKEND,scene;query_batch=1,capacity=64,
        pair_batch=1,max_iterations=1)
    g=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
    PS.queued_swept_collisions!(g,incomplete,scene)
    p=PS.Mantle.Plan(g;policy=PS.Mantle.InOrder());PS.Mantle.record!(p)
    PS.Mantle.run!(p);PS.Mantle.waitfor!(p)
    @test only(Array(scene.status)) & UInt32(2)!=0
    @test only(Array(scene.bound))==0
    @test only(Array(incomplete.state.running))!=0
    @test only(Array(incomplete.overflow))==0
    # Capacity failure must not truncate a colliding tile into a safe result;
    # the same plan recovers after geometry returns within capacity.
    bounded_scene=PS.restore_ccd_snapshot(BACKEND,snapshot)
    bounded=PS.StreamingCCDQueue(BACKEND,bounded_scene;query_batch=4,capacity=1,pair_batch=1)
    bg=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
    PS.queued_swept_collisions!(bg,bounded,bounded_scene)
    bp=PS.Mantle.Plan(bg;policy=PS.Mantle.InOrder());PS.Mantle.record!(bp)
    for (lift,overflow) in ((.02,true),(0.,false))
        finish=copy(snapshot.data.finish)
        for i in 1:4
            p=finish[i]
            finish[i]=PS.Position(PS.Vec3d(p.hi)+PS.Vec3d(p.lo)+PS.Vec3d(0,0,lift))
        end
        copyto!(bounded_scene.finish,finish)
        copyto!(bounded_scene.status,UInt32[0])
        copyto!(bounded_scene.bound,UInt32[reinterpret(UInt32,1f0)])
        for level in 0:bounded_scene.depth
            PS.launch!(BACKEND,PS.swept_refit!,(bounded_scene,Int32(level)),length(bounded_scene.left))
        end
        PS.Mantle.run!(bp);PS.Mantle.waitfor!(bp)
        @test (only(Array(bounded.overflow))>0)==overflow
        @test (only(Array(bounded_scene.status)) & UInt32(4)!=0)==overflow
        @test (only(Array(bounded_scene.bound))==0)==overflow
        @test only(Array(bounded.state.running))==0
    end
    @test_throws ArgumentError PS.StreamingCCDQueue(BACKEND,
        PS.restore_ccd_snapshot(BACKEND,snapshot);query_batch=0)
end
