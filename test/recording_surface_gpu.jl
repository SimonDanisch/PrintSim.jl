# Display reconstruction fixtures, not accepted dynamics trajectories.
using PrintSim,Test,LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "device recording display preserves affine and split-weight FE geometry" begin
    b=ModelBuilder()
    axes=(PS.Vec3d(1,0,0),PS.Vec3d(0,1,0),PS.Vec3d(0,0,1))
    add_body!(TEST_DEVICE, b,box_mesh(PS.Vec3d(.1,0,0),axes,(.01,.01,.01)))
    points,tets=tet_box((.008,.006,.002),(2,2,2))
    part=add_deformable!(b,points,tets)
    piece=b.deformables[part]
    rows=[(n,0,0,0,0,0) for n in piece.surface_nodes]
    weights=fill((1.,0.,0.,0.,0.,0.),length(rows))
    rows[1]=(1,2,3,4,5,6)
    weights[1]=PS.quadratic_triangle_weights((7001/8192,500/8192,691/8192))
    maps=Dict(piece.mesh=>(;nodes=rows,weights))
    fem=PS.FEMState(Array,b)
    body=PS.AffineState(Array,b)
    frame(t,x)=PS.DynamicsFrame(t,PS.Position[],body.q,x,PS.Point3f[],
        Float32[],UInt8[],PS.Mat3f[],Float32[],Int32[],Float32[])
    for map in (nothing,maps)
        rec=Recording(b;compact=true,surface_maps=map)
        first=frame(0.,fem.x)
        surface=PS.RecordingSurface(BACKEND,rec,first)
        @test Array(surface.points)≈PS.frame_surface(rec,first) atol=1e-8
        @test Array(surface.faces)==rec.faces
        second=frame(.001,[PS.moved(p,PS.Vec3f(1e-5,2e-5,-1e-5),1f0) for p in fem.x])
        points=surface.points
        @test PS.update_recording_surface!(surface,second)===points
        @test Array(points)≈PS.frame_surface(rec,second) atol=1e-8
        if map!==nothing
            @test eltype(surface.geometry.embedding.weight)===PS.SurfaceWeight
        end
    end
end
