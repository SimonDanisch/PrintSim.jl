using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:initial_crossing_cases) || include("initial_crossing.jl")
isdefined(@__MODULE__,:BACKEND) || error("select an explicit software or coordinated hardware backend")

function initial_pair_regression!(out,scene)
    PS.KI.get_global_id(Int32).x==1 || return nothing
    @inbounds out[1]=PS.swept_triangle_pair!(scene,Int32.((1,2,3)),Int32.((4,5,6)),0f0,true)
    return nothing
end

include("initial_separation.jl")
function initial_separated_pair_regression!(out,scene,a,b)
    PS.KI.get_global_id(Int32).x==1 || return nothing
    @inbounds out[1]=PS.swept_triangle_pair!(scene,a,b,0f0,true)
    return nothing
end

@testset "static segment/triangle separation agrees on immediate and recorded GPU" begin
    for case in captured_initial_separation_cases()
        scene=(;start=PS.upload(BACKEND,case.points),finish=PS.upload(BACKEND,case.points),refinement=Val(true),
            bound=PS.upload(BACKEND,UInt32[reinterpret(UInt32,1f0)]),status=PS.upload(BACKEND,UInt32[0]))
        out=PS.upload(BACKEND,UInt32[99])
        graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
        args=(out,scene,case.a,case.b)
        PS.launch!(graph,initial_separated_pair_regression!,args,1;group=64)
        plan=PS.Mantle.Plan(graph;policy=PS.Mantle.InOrder());PS.Mantle.record!(plan)
        for recorded in (false,true),replay in 1:2
            copyto!(out,UInt32[99])
            if recorded
                PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
            else
                PS.launch!(BACKEND,initial_separated_pair_regression!,args,1;group=64)
            end
            @test only(Array(out))==0
            @test only(Array(scene.status))==0
            @test only(Array(scene.bound))==reinterpret(UInt32,1f0)
        end
    end
end

@testset "initial edge-face rejection retains physical zero time in GPU replay" begin
    for points in initial_crossing_cases(),refinement in (Val(false),Val(true))
        scene=(;start=PS.upload(BACKEND,points),finish=PS.upload(BACKEND,points),refinement,
            bound=PS.upload(BACKEND,UInt32[reinterpret(UInt32,1f0)]),status=PS.upload(BACKEND,UInt32[0]))
        out=PS.upload(BACKEND,UInt32[0])
        graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
        PS.launch!(graph,initial_pair_regression!,(out,scene),1;group=64)
        plan=PS.Mantle.Plan(graph;policy=PS.Mantle.InOrder());PS.Mantle.record!(plan)
        for recorded in (false,true),replay in 1:2
            copyto!(scene.bound,UInt32[reinterpret(UInt32,1f0)]);copyto!(scene.status,UInt32[0]);copyto!(out,UInt32[0])
            if recorded
                PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
            else
                PS.launch!(BACKEND,initial_pair_regression!,(out,scene),1;group=64)
            end
            @test only(Array(out)) & UInt32(2)!=0
            @test only(Array(scene.status)) & UInt32(2)!=0
            @test only(Array(scene.bound))==0
        end
    end
end
