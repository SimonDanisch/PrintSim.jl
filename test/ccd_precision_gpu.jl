using PrintSim,Test
import PrintSim as PS
import Mantle
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")
isdefined(@__MODULE__,:precision_ccd_fixture) || include("ccd_precision.jl")

function precision_ccd_result!(flags,time,ccd,separation::Float32)
    PS.KI.get_global_id(Int32).x==1 || return nothing
    result=PS.swept_primitive(ccd,Int32.((1,2,3,4)),Val(false),separation,true)
    @inbounds begin
        flags[1]=result.hit ? UInt32(1) : UInt32(0)
        flags[2]=result.resolved ? UInt32(1) : UInt32(0)
        time[1]=result.time
    end
    return nothing
end

function spatial_ccd_result!(flags,ccd)
    PS.KI.get_global_id(Int32).x==1 || return nothing
    @inbounds flags[1]=PS.swept_triangle_pair!(ccd,Int32.((1,2,3)),Int32.((4,5,6)),0f0,true)
    return nothing
end

@testset "device captured idler surface separation" begin
    if Mantle.supports_float64(TEST_DEVICE)
        for thin in (false,true)
            fixture=spatial_ccd_fixture(;thin)
            ccd=map(v->v isa AbstractArray ? PS.upload(BACKEND,v) : v,fixture)
            flags=PS.upload(BACKEND,zeros(UInt32,1))
            PS.launch!(BACKEND,spatial_ccd_result!,(flags,ccd),1)
            @test only(Array(flags))==0
            @test only(Array(ccd.status))==0
        end
    else
        @test_skip false
    end
end

@testset "device precision filter preserves narrow gaps and true crossings" begin
    if Mantle.supports_float64(TEST_DEVICE)
        for (crossing,separation,hit) in ((false,0f0,false),(true,0f0,true),(false,1f-4,true))
            fixture=precision_ccd_fixture(;crossing)
            ccd=(start=PS.upload(BACKEND,fixture.start),finish=PS.upload(BACKEND,fixture.finish),refinement=Val(true))
            flags=PS.upload(BACKEND,zeros(UInt32,2))
            time=PS.upload(BACKEND,zeros(Float32,1))
            PS.launch!(BACKEND,precision_ccd_result!,(flags,time,ccd,separation),1)
            reference=PS.swept_primitive(fixture,Int32.((1,2,3,4)),Val(false),separation,true)
            # A positive conservative prefix is valid even when the spatial
            # inclusion queue remains unresolved. Preserve that diagnostic;
            # do not mistake it for a completed narrow-phase solve.
            @test Array(flags)==UInt32[reference.hit,reference.resolved] && reference.hit==hit
            @test crossing ? 0<only(Array(time))<=.5f0 : !hit || only(Array(time))==0
        end
    else
        @test_skip false
    end
end
