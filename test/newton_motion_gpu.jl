using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select an explicit backend")

@testset "CFL motion cap preserves raw Newton audit and earlier bounds" begin
    maximum_step=PS.upload(BACKEND,UInt32[0])
    bound=PS.upload(BACKEND,UInt32[reinterpret(UInt32,1f0)])
    limit=15f-6
    graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
    PS.launch!(graph,PS.newton_motion_bound!,(bound,maximum_step,limit),1;group=64)
    plan=PS.Mantle.Plan(graph;policy=PS.Mantle.InOrder());PS.Mantle.record!(plan)
    immediate=Dict{Tuple{UInt32,UInt32},UInt32}()
    for recorded in (false,true),step in (0f0,1f-10,1f-6,.003f0,Inf32,NaN32,-1f0),prior in (1f0,.01f0)
        raw=reinterpret(UInt32,step)
        copyto!(maximum_step,UInt32[raw]);copyto!(bound,UInt32[reinterpret(UInt32,prior)])
        if recorded
            PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
        else
            PS.launch!(BACKEND,PS.newton_motion_bound!,(bound,maximum_step,limit),1;group=64)
        end
        cap=reinterpret(Float32,only(Array(bound)))
        # SPIR-V Float32 division may differ from CPU division by an ulp.
        # The cap is a heuristic, not a certified collision-time calculation.
        @test isapprox(cap,min(prior,PS.newton_motion_fraction(step,limit));rtol=4eps(Float32),atol=0)
        key=(raw,reinterpret(UInt32,prior))
        if recorded
            @test reinterpret(UInt32,cap)==immediate[key]
        else
            immediate[key]=reinterpret(UInt32,cap)
        end
        @test 0<=cap<=prior
        @test only(Array(maximum_step))==raw
        if isfinite(step) && step>0
            @test Float64(cap)*Float64(step)<=Float64(limit)/8*(1+4eps(Float32))
        end
    end
    @test PS.newton_motion_fraction(.003f0,0f0)==1f0
    @test_throws ArgumentError PS.WorldSolver(nothing,PS.ModelBuilder();newton_motion_limit=-1)
    @test_throws ArgumentError PS.WorldSolver(nothing,PS.ModelBuilder();newton_motion_limit=NaN)
end
