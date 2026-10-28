using PrintSim,Test
import PrintSim as PS

struct CCDTransferWitness{T} <: AbstractVector{T}
    data::Vector{T}
    transfers::Vector{UnitRange{Int}}
end
Base.size(a::CCDTransferWitness)=size(a.data)
Base.view(::CCDTransferWitness,::AbstractUnitRange)=error("diagnostics must not materialize a device view")
function Base.copyto!(destination::Array{T},doffs::Integer,source::CCDTransferWitness{T},soffs::Integer,n::Integer) where T
    push!(source.transfers,Int(soffs):Int(soffs+n-1))
    copyto!(destination,doffs,source.data,soffs,n)
end

@testset "CCD diagnostics request only the explicit transfer range" begin
    blocking=CCDTransferWitness(zeros(Int32,10000),UnitRange{Int}[])
    reason=CCDTransferWitness(zeros(UInt32,10000),UnitRange{Int}[])
    blocking.data[9997]=2
    reason.data[9997]=3
    geometry=(;queries=fill(Int32(1),10000),faces=[Int32.(f) for f in ((1,2,3),(4,5,6))],owner=Int32[10,20])
    report=PS.ccd_failure_report((;blocking,blocking_reason=reason);query_range=9997:9998,geometry)
    @test only(report).query==9997
    @test only(report).owners==(10,20)
    @test blocking.transfers==[9997:9998]
    @test reason.transfers==[9997:9998]
end

@testset "CCD blocker diagnostics preserve and prioritize zero-time pairs" begin
    ccd=(blocking=zeros(Int32,2),blocking_reason=zeros(UInt32,2),queries=Int32[1,2],
        faces=[(Int32(1),Int32(2),Int32(3)),(Int32(4),Int32(5),Int32(6)),
            (Int32(7),Int32(8),Int32(9))],owner=Int32[10,20,30])
    PS.record_swept_blocker!(ccd,1,Int32(2),UInt32(0))
    @test ccd.blocking[1]==0
    PS.record_swept_blocker!(ccd,1,Int32(2),UInt32(1))
    PS.record_swept_blocker!(ccd,2,Int32(3),UInt32(1))
    PS.record_swept_blocker!(ccd,2,Int32(1),UInt32(3))
    PS.record_swept_blocker!(ccd,2,Int32(3),UInt32(1))
    PS.record_swept_blocker!(ccd,2,Int32(3),UInt32(2))
    @test ccd.blocking==Int32[2,1]
    @test ccd.blocking_reason==UInt32[1,3]
    report=PS.ccd_failure_report(ccd;limit=1)
    @test length(report)==1
    @test report[1].query==2 && report[1].target_face==1
    @test report[1].owners==(20,10)
    @test report[1].vertices==(ccd.faces[2],ccd.faces[1])
    @test report[1].flags==3
    @test isempty(PS.ccd_failure_report(ccd;limit=0))
    @test_throws ArgumentError PS.ccd_failure_report(ccd;limit=-1)
    @test PS.ccd_failure_report(ccd;query_range=2:2)==PS.ccd_failure_report(ccd;limit=1)
    mirror=(;queries=copy(ccd.queries),faces=copy(ccd.faces),owner=copy(ccd.owner))
    @test PS.ccd_failure_report(ccd;query_range=2:2,geometry=mirror)==report
    @test isempty(PS.ccd_failure_report(ccd;query_range=1:0))
    @test_throws ArgumentError PS.ccd_failure_report(ccd;query_range=0:1)
    @test_throws ArgumentError PS.ccd_failure_report(ccd;query_range=[1,2])
    # Zero-limit reporting must not touch any device buffer.
    @test isempty(PS.ccd_failure_report((;blocking=nothing);limit=0,query_range=1:1))
end
