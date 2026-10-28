using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:captured_projection_positive_fixture) || include("ccd_projection.jl")
isdefined(@__MODULE__,:BACKEND) || error("select an explicit software or coordinated GPU backend")

@testset "primitive-parallel CCD preserves pair flags, bounds, padding and replay" begin
    vf=captured_projection_fixture()
    ee=captured_projection_prefix_fixture()
    positive=captured_projection_positive_fixture()
    first=PS.Position.(PS.Vec3d[(0,0,0),(.01,0,0),(0,.01,0),
        (.002,.002,.001),(.008,.002,.001),(.002,.008,.001)])
    last=copy(first)
    for i in 4:6
        last[i]=PS.Position(PS.Vec3d(first[i].hi)+PS.Vec3d(first[i].lo)-PS.Vec3d(0,0,.002))
    end
    crossing=(;start=first,finish=last)
    cases=((vf,((1,2,3),(4,3,5))),
        (ee,((1,2,1),(3,4,3))),
        (positive,((1,1,1),(2,3,4))),
        (crossing,((1,2,3),(4,5,6))))
    for (case,(fixture,faces)) in enumerate(cases)
      original=nothing
      for track_minimum in (false,true)
        reference=nothing
        for parallel in (false,true),recorded in (false,true)
            println((ccd_parallel_fixture=case,parallel,recorded,stage=:prepare));flush(stdout)
            scene=(;start=PS.upload(BACKEND,fixture.start),finish=PS.upload(BACKEND,fixture.finish),
                faces=PS.upload(BACKEND,[Int32.(face) for face in faces]),queries=PS.upload(BACKEND,Int32[1]),
                owner=PS.upload(BACKEND,Int32[1,2]),radius=PS.upload(BACKEND,Float32[0,0]),
                refinement=Val(true),bound=PS.upload(BACKEND,UInt32[reinterpret(UInt32,1f0)]),
                status=PS.upload(BACKEND,UInt32[0]))
            work=(;pairs=PS.upload(BACKEND,fill((Int32(1),Int32(2)),10)),
                flags=PS.upload(BACKEND,fill(UInt32(127),10)))
            if track_minimum
                work=merge(work,(;pair_bound=PS.upload(BACKEND,fill(reinterpret(UInt32,1f0),10)),
                    minimum=PS.upload(BACKEND,UInt32[reinterpret(UInt32,1f0),0,0])))
            end
            total=PS.upload(BACKEND,Int32[7])
            iteration=PS.upload(BACKEND,Int32[1])
            graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
            PS.launch_ccd_pair_checks!(graph,work,scene,total,iteration,3;primitive_parallel=parallel)
            plan=PS.Mantle.Plan(graph;policy=PS.Mantle.InOrder());PS.Mantle.record!(plan)
            println((ccd_parallel_fixture=case,parallel,recorded,stage=:recorded));flush(stdout)
            for _ in 1:2
                copyto!(scene.bound,UInt32[reinterpret(UInt32,1f0)]);copyto!(scene.status,UInt32[0])
                copyto!(work.flags,fill(UInt32(127),10))
                track_minimum && copyto!(work.minimum,UInt32[reinterpret(UInt32,1f0),0,0])
                for part in 1:3
                    println((ccd_parallel_fixture=case,parallel,recorded,stage=:execute,part));flush(stdout)
                    copyto!(iteration,Int32[part])
                    if recorded
                        PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
                    else
                        PS.launch_ccd_pair_checks!(BACKEND,work,scene,total,iteration,3;primitive_parallel=parallel)
                    end
                end
                actual=(;flags=Array(work.flags),bound=Array(scene.bound),status=Array(scene.status),
                    minimum=track_minimum ? Array(work.minimum) : nothing)
                reference===nothing && (reference=actual)
                original===nothing && (original=actual)
                @test actual.flags==original.flags && actual.bound==original.bound && actual.status==original.status
                @test actual.flags==reference.flags
                @test actual.bound==reference.bound
                @test actual.status==reference.status
                @test actual.flags[8:10]==fill(UInt32(127),3)
                if track_minimum
                    @test actual.minimum==reference.minimum
                    @test actual.minimum[1]==only(actual.bound)
                    @test Array(work.pair_bound)[8:10]==fill(reinterpret(UInt32,1f0),3)
                    @test actual.minimum[2:3]==(only(actual.bound)<reinterpret(UInt32,1f0) ? UInt32[1,2] : UInt32[0,0])
                end
            end
            @test_throws ArgumentError PS.launch_ccd_pair_checks!(BACKEND,work,scene,total,iteration,0)
            @test_throws ArgumentError PS.launch_ccd_pair_checks!(BACKEND,work,scene,total,iteration,typemax(Int32))
        end
      end
    end
end
