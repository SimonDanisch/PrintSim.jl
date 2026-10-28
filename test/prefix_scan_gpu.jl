using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "recordable integer prefix scan retains all queue entries" begin
    for n in (0,1,255,256,257,65537)
        values=Int32[iszero(i%7) ? typemax(Int32) : i%11 for i in 1:n]
        source=PS.upload(BACKEND,values)
        scan=PS.DevicePrefixScan(BACKEND,n)
        PS.prefix_scan!(BACKEND,scan,source)
        expected=cumsum(Int64.(values))
        offsets=n==0 ? Int64[] : [0;expected[1:end-1]]
        @test Array(scan.offsets[1])[1:n]==offsets
        @test only(Array(scan.totals[end]))==sum(Int64,values;init=0)
    end
    source=PS.upload(BACKEND,Int32[3,0,5,1])
    scan=PS.DevicePrefixScan(BACKEND,4)
    graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
    PS.prefix_scan!(graph,scan,source)
    plan=PS.Mantle.Plan(graph)
    PS.Mantle.record!(plan)
    for values in (Int32[3,0,5,1],Int32[1,2,0,7])
        copyto!(source,values)
        PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
        @test Array(scan.offsets[1])==[0;cumsum(Int64.(values))[1:end-1]]
        @test only(Array(scan.totals[end]))==sum(values)
    end
    @test_throws ArgumentError PS.DevicePrefixScan(BACKEND,-1)
    @test_throws DimensionMismatch PS.prefix_scan!(BACKEND,scan,Int32[1])
end
