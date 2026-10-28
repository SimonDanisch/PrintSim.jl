using PrintSim,Test,LinearAlgebra
import PrintSim as PS

@testset "flat quadratic boundary need not be uniformly oversampled" begin
    b=ModelBuilder()
    points,tets=tet_box((.001,.001,.001),(2,2,2))
    add_deformable!(b,points,tets)
    piece=only(b.deformables)
    parents=PS.quadratic_boundary_faces(piece)
    proxy=PS.quadratic_surface_proxy(piece.x,parents;minimum_rounds=0,tolerance=1e-9)
    @test length(proxy.faces)==length(parents)
    @test proxy.rounds==0
    @test proxy.max_error<1e-12
    @test all(i->PS.quadratic_proxy_oriented(parents[proxy.parent[i]],proxy.parameters[i],
        proxy.faces[i],proxy.points,piece.x),eachindex(proxy.faces))
    @test PS.closed_surface_topology(proxy.points,proxy.faces)==(components=1,euler=2)
    @test_throws ArgumentError PS.quadratic_surface_proxy(piece.x,parents;minimum_rounds=-1)
end

@testset "canonical boundary keys retain ordering without temporary vectors" begin
    for f in ((1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1),(1,1,2),(-3,2,-1))
        @test PS.triangle_node_key(f)==Tuple(sort(collect(f)))
    end
end

@testset "uncertain skinny-facet orientation has an exact fallback" begin
    rest=PS.Vec3d[(0,0,0),(0,1,0),(0,1,1e-24),
        (0,.5,0),(0,.5,5e-25),(0,1,5e-25)]
    nodes=(1,2,3,4,5,6)
    uv=(PS.Vec3d(1,0,0),PS.Vec3d(0,1,0),PS.Vec3d(0,0,1))
    @test PS.quadratic_proxy_oriented_exact(nodes,uv,(1,2,3),rest,rest)
    @test PS.quadratic_proxy_oriented(nodes,uv,(1,2,3),rest,rest)
    @test !PS.quadratic_proxy_oriented_exact(nodes,uv,(1,3,2),rest,rest)
    @test !PS.quadratic_proxy_oriented(nodes,uv,(1,3,2),rest,rest)
end

@testset "thin printed seam terminates with oriented conforming refinement" begin
    # A real right-idler boundary face: 11 mm long, about 15 nm wide.
    # Arbitrarily repeating green templates used to move its reversed chords
    # between children instead of resolving them.
    rest=PS.Vec3d[(.12,-.024231300039701975,.05959295804190372),
        (.12,-.013051200167963042,.05959211189197213),
        (.12,-.024442478876589704,.05959295925685296),
        (.12,-.01864125010383251,.05959253496693792),
        (.12,-.02433688945814584,.059592958649378344),
        (.12,-.018746839522276373,.05959253557441255)]
    initial=[p+PS.Vec3d(-5e-9,0,-(p[2]+.0132)^2) for p in rest]
    parent=(Int32(1),Int32(2),Int32(3),Int32(4),Int32(5),Int32(6))
    proxy=PS.quadratic_surface_proxy(rest,[parent];initial,tolerance=1e-8,
        max_rounds=40,max_faces=100_000)
    @test proxy.max_error<=1e-8
    @test all(t->PS.quadratic_proxy_oriented(parent,proxy.parameters[t],proxy.faces[t],proxy.initial,initial),eachindex(proxy.faces))
    @test all(t->PS.quadratic_proxy_oriented(parent,proxy.parameters[t],proxy.faces[t],proxy.points,rest),eachindex(proxy.faces))
    captured=Ref(false)
    @test_throws ErrorException PS.quadratic_surface_proxy(rest,[parent];initial,
        tolerance=1e-8,max_rounds=1,on_failure=x->(captured[]=!isempty(x.marked)))
    @test captured[]
end

@testset "distance tolerance alone must not admit folded proxy facets" begin
    corners=PS.Vec3d[(0,0,0),(0,1,0),(0,2,.001)]
    rest=[corners;[(corners[1]+corners[2])/2,(corners[1]+corners[3])/2,(corners[2]+corners[3])/2]]
    initial=[p+PS.Vec3d(0,0,-.1p[2]^2) for p in rest]
    faces=[(Int32(1),Int32(2),Int32(3),Int32(4),Int32(5),Int32(6))]
    # This is an injective shear, but coarse chords reverse the skinny
    # triangle. Loose distance tolerance deliberately cannot trigger the repair.
    proxy=PS.quadratic_surface_proxy(rest,faces;initial,tolerance=1.,max_faces=100000,max_rounds=10)
    @test proxy.oriented_facets
    @test length(proxy.faces)>4
    @test all(i->PS.quadratic_proxy_oriented(faces[1],proxy.parameters[i],proxy.faces[i],proxy.initial,initial),eachindex(proxy.faces))
    @test all(i->PS.quadratic_proxy_oriented(faces[1],proxy.parameters[i],proxy.faces[i],proxy.points,rest),eachindex(proxy.faces))
end

@testset "adaptive high-order collision proxy is conforming and error-bounded" begin
    rest=PS.Vec3d[(0,0,0),(1,0,0),(0,1,0),(0,0,1),
        (.5,0,0),(0,.5,0),(0,0,.5),(.5,.5,0),(.5,0,.5),(0,.5,.5)]
    faces=[Int32.(f) for f in [(2,3,4,8,9,10),(1,4,3,7,6,10),
        (1,2,4,5,7,9),(1,3,2,6,5,8)]]
    # A globally injective quadratic shear, like the idler preload.
    initial=[p+PS.Vec3d(0,0,.1p[2]^2) for p in rest]
    proxy=PS.quadratic_surface_proxy(rest,faces;initial,tolerance=.001,max_rounds=8)
    @test length(proxy.faces)>16
    @test proxy.max_error<=proxy.tolerance
    edge_counts=Dict{Tuple{Int32,Int32},Int}()
    edge_orient=Dict{Tuple{Int32,Int32},Int}()
    for f in proxy.faces,k in 1:3
        a,b=f[k],f[k==3 ? 1 : k+1]
        edge=minmax(a,b)
        edge_counts[edge]=get(edge_counts,edge,0)+1
        edge_orient[edge]=get(edge_orient,edge,0)+(a<b ? 1 : -1)
    end
    @test all(==(2),values(edge_counts))
    @test all(iszero,values(edge_orient))
    @test length(proxy.points)-length(edge_counts)+length(proxy.faces)==2
    volume=sum(dot(proxy.points[f[1]],cross(proxy.points[f[2]],proxy.points[f[3]]))/6 for f in proxy.faces)
    @test volume≈1/6 atol=1e-14
    map=PS.SurfaceEmbedding(Array,proxy.nodes,proxy.weights,10)
    @test all(i->PS.surface_map_motion(map,i,initial)≈proxy.initial[i],eachindex(proxy.initial))
    # Interior quadratic samples must remain within the advertised facet error.
    for i in eachindex(proxy.faces)
        f,uv,p=proxy.faces[i],proxy.parameters[i],proxy.parent[i]
        for bary in ((.17,.23,.60),(.3,.3,.4),(.1,.8,.1))
            parameter=sum(bary[k]*uv[k] for k in 1:3)
            weights=PS.quadratic_triangle_weights(parameter)
            exact=sum(weights[k]*initial[faces[p][k]] for k in 1:6)
            linear=sum(bary[k]*proxy.initial[f[k]] for k in 1:3)
            @test norm(exact-linear)<=proxy.max_error+1e-14
        end
    end
    flat=PS.quadratic_surface_proxy(rest,faces;tolerance=1e-12)
    @test length(flat.faces)==16 && length(flat.points)==10
    @test_throws ErrorException PS.quadratic_surface_proxy(rest,faces;initial,tolerance=1e-12,max_rounds=1)
    @test_throws ErrorException PS.quadratic_surface_proxy(rest,faces;initial,max_faces=15)
    @test_throws ArgumentError PS.quadratic_surface_proxy(rest,faces;tolerance=0.)
end
