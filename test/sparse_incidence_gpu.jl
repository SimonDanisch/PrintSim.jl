using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || error("select a coordinated GPU or explicit software backend")

@testset "sparse contact incidence matches sorted dense rows on replay" begin
    for (entries,nodes,capacity) in ((0,0,4),(1,7,4),(777,19,256),(777,19,3))
        sparse=PS.SparseVertexIncidence(BACKEND,entries,nodes;capacity)
        @test length(sparse.ids.data)==max(1,4entries)
        keys=zeros(Int32,4,max(1,entries))
        for entry in 1:entries
            keys[:,entry].=Int32[mod(entry,5)==0 ? 0 : mod1(entry,nodes),
                mod1(entry+3,nodes),mod1(entry+3,nodes),0]
        end
        graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
        PS.compact!(graph,sparse)
        plan=PS.Mantle.Plan(graph;policy=PS.Mantle.InOrder())
        PS.Mantle.record!(plan)
        for replay in 1:2
            replay==2 && (keys[:,1:entries].=0) # stale rows/counts may not survive replay
            copyto!(sparse.keys,keys)
            PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
            count,overflow=Array(sparse.count),Array(sparse.overflow)
            data,offset=Array(sparse.ids.data),Array(sparse.ids.offset)
            for node in 1:nodes
                expected=Int32[e for e in 1:entries if node in keys[:,e]]
                @test count[node]==min(length(expected),capacity)
                @test overflow[node]==max(0,length(expected)-capacity)
                if length(expected)<=capacity
                    @test data[offset[node].+(1:count[node])]==expected
                end
            end
        end
    end
    @test_throws ArgumentError PS.SparseVertexIncidence(BACKEND,-1,3)
    @test_throws ArgumentError PS.SparseVertexIncidence(BACKEND,1,3;capacity=0)
end
