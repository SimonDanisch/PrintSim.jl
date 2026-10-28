using PrintSim,Test
import PrintSim as PS

@testset "surface and proxy hot arithmetic has concrete inferred results" begin
    rows=[Int32.((1,2,3,4,5,6))]
    p=PS.Vec3f[PS.Vec3f(i/10,i/20,i/30) for i in 1:6]
    x=PS.Position.(PS.Vec3d.(p))
    for bary in ((.5,.25,.25),(7001/8192,500/8192,691/8192))
        map=PS.SurfaceEmbedding(Array,rows,[PS.quadratic_triangle_weights(bary)],6)
        @test (@inferred PS.surface_map_motion(map,Int32(1),p)) isa PS.Vec3f
        @test (@inferred PS.surface_map_force(map,Int32(1),p)) isa PS.Vec3f
        @test (@inferred PS.surface_map_position(map,Int32(1),x,1f0)) isa PS.Position
    end
    guard=(;nodes=reshape(Int32.(1:6),6,1),parent=Int32[1],
        edges=fill(PS.Vec3f(.5,-.5,0),3,1))
    q=@inferred PS.proxy_parent_curvature(guard,Int32(1),x,1f0)
    @test q isa PS.ProxyCurvature
    @test (@inferred PS.proxy_facet_error(guard,Int32(1),[q],1f0)) isa Float32
    soa=(;ab=[q.q[1]],ac=[q.q[2]],bc=[q.q[3]],error=[q.error])
    @test (@inferred PS.proxy_facet_error(guard,Int32(1),soa,1f0)) isa Float32
end
