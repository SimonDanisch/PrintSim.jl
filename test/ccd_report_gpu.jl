using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "bounded CCD device readback uses a host geometry mirror" begin
    geometry=(queries=Int32[1,2],faces=[Int32.(f) for f in [(1,2,3),(4,5,6),(7,8,9)]],owner=Int32[10,20,30])
    ccd=(blocking=PS.upload(BACKEND,Int32[2,1]),blocking_reason=PS.upload(BACKEND,UInt32[1,3]),
        queries=PS.upload(BACKEND,geometry.queries),faces=PS.upload(BACKEND,geometry.faces),
        owner=PS.upload(BACKEND,geometry.owner))
    report=PS.ccd_failure_report(ccd;query_range=2:2,geometry)
    @test length(report)==1 && report[1].query==2
    @test report[1].owners==(20,10) && report[1].flags==3
    @test PS.ccd_failure_report(ccd;limit=1)==report
end

@testset "bounded CCD readback at a large device offset" begin
    n=4_095_744
    values=zeros(Int32,n)
    values[end-1:end].=Int32[17,29]
    device_values=PS.upload(BACKEND,values)
    @test PS.ccd_host_range(device_values,n-1:n)==Int32[17,29]
    @test PS.ccd_host_range(device_values,1:2)==Int32[0,0]
end
