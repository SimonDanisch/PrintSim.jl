using PrintSim,Test,LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "device proxy bound agrees with host and rejects nonfinite curvature" begin
    b=ModelBuilder()
    points,tets=tet_box((.008,.006,.002),(2,2,2))
    part=add_deformable!(b,points,tets;place=p->p+PS.Vec3d(.1p[2]^2,.2p[1]*p[3],.3p[1]^2))
    piece=b.deformables[part]
    proxy=PS.quadratic_surface_proxy(piece.x,PS.quadratic_boundary_faces(piece);initial=piece.initial,tolerance=1e-7)
    maps=PS.install_surface_proxies!(b,Dict(part=>proxy))
    host=PS.QuadraticProxyGuard(Array,b,maps)
    device=PS.QuadraticProxyGuard(BACKEND,b,maps)
    x=PS.Position.(piece.initial)
    one=PS.upload(BACKEND,Float32[1])
    expected=[PS.proxy_parent_curvature(host,i,x,1f0) for i in 1:host.parents]
    PS.launch!(BACKEND,PS.proxy_curvature!,(device,PS.upload(BACKEND,x),PS.upload(BACKEND,Float32[1]),Int32(0),Int32(host.parents)),host.parents)
    actual_curvature=(Array(device.curvature.ab),Array(device.curvature.ac),Array(device.curvature.bc))
    precise(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
    @test maximum(norm(precise(actual_curvature[k][i])-precise(expected[i].q[k])) for i in eachindex(expected),k in 1:3)<1e-14
    for count in (1,255,256,257,host.faces)
        count<=host.faces || continue
        PS.launch!(BACKEND,PS.zero!,(device.maximum,),1)
        PS.launch!(BACKEND,PS.proxy_error_bound!,(device,one,Int32(0),Int32(count)),count)
        values=reinterpret(Float32,Array(view(device.maximum,2:count+1)))
        reference=[PS.proxy_facet_error(host,f,expected) for f in 1:count]
        @test values≈reference rtol=2e-5
        PS.launch!(BACKEND,PS.proxy_reduce_error!,(device,Int32(0),Int32(count)),count)
        actual=reinterpret(Float32,only(Array(view(device.maximum,1:1))))
        bound=maximum(reference)
        @test actual≈bound rtol=2e-5
    end
    invalid=PS.upload(BACKEND,fill(PS.Position(PS.Vec3f(NaN)),device.parents))
    bad_curvature=(ab=invalid,ac=invalid,bc=invalid,error=PS.upload(BACKEND,fill(Inf32,device.parents)))
    bad=PS.QuadraticProxyGuard(device.nodes,device.parent,device.edges,bad_curvature,device.maximum,device.parents,device.faces)
    PS.launch!(BACKEND,PS.proxy_error_bound!,(bad,one,Int32(0),Int32(bad.faces)),bad.faces)
    @test all(==(Inf32),reinterpret(Float32,Array(view(bad.maximum,2:bad.faces+1))))
    PS.launch!(BACKEND,PS.proxy_reduce_error!,(bad,Int32(0),Int32(bad.faces)),bad.faces)
    @test reinterpret(Float32,only(Array(view(bad.maximum,1:1))))==Inf32
end
