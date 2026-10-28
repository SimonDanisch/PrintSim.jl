using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

function batch_increment!(out,first::Int32,count::Int32)
    lane=PS.KI.get_global_id(Int32).x
    lane<=count || return nothing
    @inbounds out[first+lane-Int32(1)]+=Int32(1)
    return nothing
end

@testset "bounded dispatch covers each query exactly once including padded tails" begin
    for n in (0,1,257,1031),batch in (3,256,8192)
        out=PS.upload(BACKEND,zeros(Int32,max(n,1)))
        PS.launch_indexed_batches!(BACKEND,batch_increment!,(out,),n;batch)
        @test Array(out)==fill(Int32(n>0),max(n,1))
    end
    @test_throws ArgumentError PS.launch_indexed_batches!(BACKEND,batch_increment!,(),1;batch=0)
end
