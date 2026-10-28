using PrintSim,Test,LinearAlgebra
import PrintSim as PS

@testset "P2 collision embedding reproduces quadratic fields and their adjoint" begin
    corners=PS.Vec3d[(0,0,0),(1,0,0),(0,1,0),(.5,0,0),(0,.5,0),(.5,.5,0)]
    bary=[(i/8,j/8,(8-i-j)/8) for i in 0:8 for j in 0:8-i]
    rows=fill(Int32.((1,2,3,4,5,6)),length(bary))
    weights=PS.quadratic_triangle_weights.(bary)
    embedding=PS.SurfaceEmbedding(Array,rows,weights,6)
    W=zeros(Float64,length(bary),6)
    for row in eachindex(bary),k in 1:6
        node=embedding.node[row,k]
        node==0 || (W[row,node]=embedding.weight[row,k])
    end
    @test W*ones(6)==ones(length(bary))
    field(p)=PS.Vec3d(.02p[1]^2+.03p[1]*p[2]-.04p[2]^2,p[1]+2p[2],1)
    nodal=field.(corners)
    for i in eachindex(bary)
        point=bary[i][1]*corners[1]+bary[i][2]*corners[2]+bary[i][3]*corners[3]
        @test PS.surface_map_motion(embedding,i,nodal)≈field(point) atol=1e-14
    end
    force=[PS.Vec3d(sin(i),cos(i),sin(2i)) for i in eachindex(bary)]
    p=[PS.Vec3d(cos(i),sin(2i),cos(3i)) for i in 1:6]
    Wp=[PS.surface_map_motion(embedding,i,p) for i in eachindex(bary)]
    Wtf=[PS.surface_map_force(embedding,i,force) for i in 1:6]
    @test sum(dot.(Wp,force))≈sum(dot.(p,Wtf)) atol=1e-13
    @test sum(Wtf)≈sum(force) atol=1e-13
    # Matrix-free W' H W retains off-diagonal FE-node coupling, not just w²H.
    H=[PS.Mat3f(2,0,0,0,3,0,0,0,4) for _ in bary]
    HWp=[H[i]*Wp[i] for i in eachindex(bary)]
    action=[PS.surface_map_force(embedding,i,HWp) for i in 1:6]
    reference=transpose(W)*permutedims(reduce(hcat,HWp))
    @test permutedims(reduce(hcat,action))≈reference atol=1e-12
    @test sum(dot.(p,action))>=0
    for node in 1:6
        @test PS.surface_map_diagonal(embedding,node,H)≈sum(W[i,node]^2*H[i] for i in eachindex(bary)) atol=1e-12
    end
    x=PS.Position.(corners .+ Ref(PS.Vec3d(.123,.234,.345)))
    for i in eachindex(bary)
        actual=PS.surface_map_position(embedding,i,x,1f0)
        precise=PS.Vec3d(actual.hi)+PS.Vec3d(actual.lo)
        expected=sum(weights[i][j]*(corners[j]+PS.Vec3d(.123,.234,.345)) for j in 1:6)
        @test precise≈expected atol=2e-14
    end
    @test_throws ArgumentError PS.SurfaceEmbedding(Array,rows,weights,5)
    @test_throws ArgumentError PS.SurfaceEmbedding(Array,[rows[1]],[(NaN,0,0,0,0,0)],6)
    @test_throws ArgumentError PS.SurfaceEmbedding(Array,[rows[1]],[(.5,0,0,0,0,0)],6)
    @test_throws ArgumentError PS.SurfaceEmbedding(Array,[(1,1,0,0,0,0)],[(.5,.5,0,0,0,0)],6)
    empty=PS.SurfaceEmbedding(Array,NTuple{6,Int32}[],NTuple{6,Float32}[],0)
    @test empty.points==empty.nodes==0 && empty.first==Int32[1]
end

@testset "deep dyadic surface weights retain the same precise map and adjoint" begin
    bary=[(7001/8192,500/8192,691/8192),
        (1/32768,17003/32768,15764/32768),(27001/32768,2500/32768,3267/32768)]
    rows=fill(Int32.((1,2,3,4,5,6)),length(bary))
    weights=PS.quadratic_triangle_weights.(bary)
    map=PS.SurfaceEmbedding(Array,rows,weights,6)
    @test eltype(map.weight)===PS.SurfaceWeight
    @test all(i->sum(Float64,@view(map.weight[i,:]))==1,eachindex(bary))
    corners=PS.Vec3d[(0,0,0),(0,.001,0),(0,.002,1e-9)]
    points=[corners;[(corners[1]+corners[2])/2,(corners[1]+corners[3])/2,(corners[2]+corners[3])/2]]
    points=[p+PS.Vec3d(.12,0,-.1p[2]^2) for p in points]
    x=PS.Position.(points)
    for i in eachindex(bary)
        position=PS.surface_map_position(map,i,x,1f0)
        @test PS.Vec3d(position.hi)+PS.Vec3d(position.lo)≈sum(weights[i][k]*points[k] for k in 1:6) atol=1e-15
    end
    p=[PS.Vec3d(sin(i),cos(i),sin(2i)) for i in 1:6]
    f=[PS.Vec3d(cos(i),sin(i),cos(2i)) for i in eachindex(bary)]
    forward=[PS.surface_map_motion(map,i,p) for i in eachindex(bary)]
    transpose=[PS.surface_map_force(map,i,f) for i in 1:6]
    @test sum(dot.(forward,f))≈sum(dot.(p,transpose)) atol=1e-13
    @test sum(transpose)≈sum(f) atol=1e-13
end
