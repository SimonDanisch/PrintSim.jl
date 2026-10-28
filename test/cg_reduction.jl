using PrintSim,Test,LinearAlgebra
import PrintSim as PS
import Mantle
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "coupled dot reduction has a fixed cross-block order" begin
    for n in (0,1,257,1025,131073)
        a=[PS.Vec3f(sin(i/7),cos(i/11),sin(i/13)) for i in 1:n]
        b=[PS.Vec3f(cos(i/5),sin(i/17),cos(i/19)) for i in 1:n]
        da,db=PS.upload(BACKEND,a),PS.upload(BACKEND,b)
        blocks=Int32(max(1,cld(n,256)))
        partial=PS.upload(BACKEND,zeros(Float32,blocks))
        scalar=PS.upload(BACKEND,zeros(Float32,1))
        graph=Mantle.Graph(Mantle.todevice(BACKEND))
        PS.launch!(graph,PS.zero!,(scalar,),1)
        PS.launch!(graph,PS.dot_into!,(partial,da,db,Int32(n),Int32(1),Val(1)),Int(blocks)*256)
        PS.launch!(graph,PS.dot_into!,(scalar,partial,nothing,blocks,Int32(1),Val(2)),256)
        plan=Mantle.Plan(graph)
        Mantle.record!(plan)
        results=Float32[]
        for _ in 1:4
            Mantle.run!(plan);Mantle.waitfor!(plan)
            push!(results,only(Array(scalar)))
        end
        @test all(x->reinterpret(UInt32,x)==reinterpret(UInt32,first(results)),results)
        expected=sum(i->dot(Float64.(a[i]),Float64.(b[i])),1:n;init=0.0)
        @test first(results)≈expected rtol=3e-5 atol=2e-4
    end
end
