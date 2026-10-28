using PrintSim,Test,LinearAlgebra
import PrintSim as PS

@testset "deforming quadratic proxy bound covers facet interiors" begin
    b=ModelBuilder()
    points,tets=tet_box((.008,.006,.002),(2,2,2))
    part=add_deformable!(b,points,tets;place=p->p+PS.Vec3d(.1p[2]^2,.2p[1]*p[3],.3p[1]^2))
    piece=b.deformables[part]
    faces=PS.quadratic_boundary_faces(piece)
    proxy=PS.quadratic_surface_proxy(piece.x,faces;initial=piece.initial,tolerance=1e-7)
    maps=PS.install_surface_proxies!(b,Dict(part=>proxy))
    guard=PS.QuadraticProxyGuard(Array,b,maps)
    @test guard.parents==length(faces)
    @test guard.faces==length(proxy.faces)
    function check_state(x)
        precise=PS.Position.(x)
        q=[PS.proxy_parent_curvature(guard,p,precise,1f0) for p in 1:guard.parents]
        bounds=[PS.proxy_facet_error(guard,f,q) for f in 1:guard.faces]
        for f in eachindex(proxy.faces)
            uv=proxy.parameters[f]
            nodes=faces[proxy.parent[f]]
            p=[sum(PS.quadratic_triangle_weights(u)[k]*x[nodes[k]] for k in 1:6) for u in uv]
            for bary in ((.2,.3,.5),(.1,.1,.8),(.333,.333,.334))
                parameter=sum(bary[k]*uv[k] for k in 1:3)
                exact=sum(PS.quadratic_triangle_weights(parameter)[k]*x[nodes[k]] for k in 1:6)
                linear=sum(bary[k]*p[k] for k in 1:3)
                @test norm(exact-linear)<=bounds[f]+1e-16
            end
        end
        geometric=[PS.ProxyCurvature(p.q,0f0) for p in q]
        nominal=maximum(PS.proxy_facet_error(guard,f,geometric) for f in 1:guard.faces)
        # The certified bound intentionally includes absolute-coordinate
        # rounding, which is not translation invariant. Compare curvature
        # separately and retain that allowance when comparing certificates.
        allowance=maximum(begin
            uv=guard.edges[:,f]
            weights=maximum(sum(abs(d[i]*d[j]) for (i,j) in ((1,2),(1,3),(2,3))) for d in uv)
            (4/3)*weights*q[guard.parent[f]].error*(1+64eps(Float32))
        end for f in 1:guard.faces)
        return (;bound=maximum(bounds),nominal,allowance)
    end
    initial=check_state(piece.initial)
    warped=[piece.x[i]+4*(piece.initial[i]-piece.x[i]) for i in eachindex(piece.x)]
    changed_state=check_state(warped)
    changed=changed_state.bound
    @test changed>3.9initial.bound
    translated=check_state(warped .+ Ref(PS.Vec3d(.123,.234,.345)))
    @test translated.nominal≈changed_state.nominal atol=1e-16 rtol=1e-7
    @test abs(translated.bound-changed)<=translated.allowance+changed_state.allowance+1e-16
    invalid=PS.ProxyCurvature(ntuple(_->PS.Position(PS.Vec3f(NaN)),3),Inf32)
    @test PS.proxy_facet_error((parent=Int32[1],edges=guard.edges),1,[invalid])==Inf32
end

@testset "proxy curvature rounding allowance covers leading-bin crossings" begin
    boundary=Float64(.1f0)+Float64(eps(.1f0))/2
    a=PS.Vec3d(boundary-1e-12,.234,.345)
    b=PS.Vec3d(boundary+1e-12,.234,.345)
    q,error=PS.proxy_edge_curvature(PS.Position(a),PS.Position(b),PS.Position((a+b)/2),1f0)
    @test norm(PS.Vec3d(q.hi)+PS.Vec3d(q.lo))<=error
    @test error<1e-10
end

@testset "proxy bound preserves skinny-parent curvature cancellation" begin
    a,b,c=PS.Vec3d(0,0,0),PS.Vec3d(.01,.01,0),PS.Vec3d(.005001,.005,0)
    rest=[a,b,c,(a+b)/2,(a+c)/2,(b+c)/2]
    initial=[p+PS.Vec3d(0,0,p[2]^2) for p in rest]
    # Edge from reference midpoint ab to c: huge parent curvatures cancel
    # exactly because its physical y coordinate is constant.
    d=PS.Vec3f(.5,.5,-1)
    guard=(nodes=reshape(Int32.(1:6),6,1),parent=Int32[1],edges=reshape(fill(d,3),3,1))
    q=PS.proxy_parent_curvature(guard,1,PS.Position.(initial),1f0)
    bound=PS.proxy_facet_error(guard,1,[q])
    @test bound<1e-12
    @test bound>=0
    scalar_bound=(4/3)*sum(abs(d[i]*d[j])*norm(PS.Vec3d(q.q[k].hi)+PS.Vec3d(q.q[k].lo)) for (k,(i,j)) in enumerate(((1,2),(1,3),(2,3))))
    @test scalar_bound>1e-6
    @test bound<scalar_bound/1e6
end
