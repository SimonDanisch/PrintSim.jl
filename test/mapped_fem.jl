using PrintSim, Test, LinearAlgebra
import PrintSim as PS

@testset "proxy transfer preserves FE part ownership and legacy surfaces" begin
    b=ModelBuilder()
    points,tets=tet_box((.008,.006,.002),(2,2,2))
    add_deformable!(b,points,tets)
    add_deformable!(b,points,tets;place=p->p+PS.Vec3d(.02,0,0))
    data=PS.surface_transfer_data(b,Dict())
    @test data.embedding.nodes==sum(length(p.x) for p in b.deformables)
    @test data.embedding.points==sum(length(p.surface_nodes) for p in b.deformables)
    transfer=PS.FEMSurfaceTransfer(Array,b,Dict())
    base=PS.FEMModel(Array,b)
    model=PS.MappedFEMModel(base,transfer)
    state=PS.FEMState(Array,b)
    @test model.nodes==base.nodes
    @test model.mass===base.mass
    @test model.surface_node===transfer.point_row
    for point in eachindex(base.surface_node)
        @test PS.fem_surface_position(state,model,point)==state.x[base.surface_node[point]]
        @test PS.fem_surface_slip(model,state,state.x,point,1f0)==zero(PS.Vec3f)
    end
    # A P2 row with negative corner weights exercises all six slots and the
    # second part's global FE offset, rather than accidentally its neighbour.
    piece=b.deformables[2]
    rows=[(n,0,0,0,0,0) for n in piece.surface_nodes]
    weights=fill((1f0,0f0,0f0,0f0,0f0,0f0),length(rows))
    rows[1]=(1,2,3,4,5,6)
    weights[1]=Float32.(PS.quadratic_triangle_weights((.5,.25,.25)))
    maps=Dict(piece.mesh=>(nodes=rows,weights=weights))
    quadratic=PS.FEMSurfaceTransfer(Array,b,maps)
    qmodel=PS.MappedFEMModel(base,quadratic)
    point=length(b.meshes[b.deformables[1].mesh].vertices)+1
    offset=length(b.deformables[1].x)
    expected=sum(weights[1][k]*piece.initial[rows[1][k]] for k in 1:6)
    actual=PS.fem_surface_position(state,qmodel,point)
    @test PS.Vec3d(actual.hi)+PS.Vec3d(actual.lo)≈expected atol=1e-14
    row=quadratic.point_row[point]
    @test all(n->iszero(n)||n>offset,quadratic.embedding.node[row,:])
    @test_throws ArgumentError PS.surface_transfer_data(b,Dict(999=>maps[piece.mesh]))
    bad=copy(rows); bad[1]=(1,length(piece.x)+1,3,4,5,6)
    @test_throws ArgumentError PS.surface_transfer_data(b,Dict(piece.mesh=>(nodes=bad,weights=weights)))
    @test_throws ArgumentError PS.surface_transfer_data(b,Dict(piece.mesh=>(nodes=rows,weights=weights[2:end])))

    rec=Recording(b;compact=true,surface_maps=maps)
    @test rec isa PS.MappedRecording
    @test all(iszero,rec.fem_node)
    frame=PS.DynamicsFrame(0.,PS.Position[],PS.Position[],state.x,PS.Point3f[],Float32[],UInt8[],PS.Mat3f[],Float32[],Int32[],Float32[])
    displayed=PS.frame_surface(rec,frame)
    @test displayed[point]≈PS.Point3f(expected) atol=1e-8
    push!(rec.frames,frame)
    @test PS.recording_times(rec)==[0.]
    mktempdir() do dir
        path=save_recording(joinpath(dir,"mapped.jls"),rec)
        loaded=load_recording(path)
        @test loaded isa PS.MappedRecording
        @test PS.frame_surface(loaded,loaded.frames[1])==displayed
        archived=Recording(b;compact=true,surface_maps=maps,directory=dir)
        push!(archived.frames,frame)
        archive_path=save_recording(joinpath(dir,"mapped-index.jls"),archived;external_geometry=true)
        archived_loaded=load_recording(archive_path)
        @test archived_loaded isa PS.MappedRecording
        @test PS.frame_surface(archived_loaded,archived_loaded.frames[1])==displayed
        @test filesize(archive_path)<16_384
        legacy=Recording(b;compact=true)
        path=save_recording(joinpath(dir,"legacy.jls"),legacy)
        @test load_recording(path) isa Recording
    end
end

@testset "proxy installation retains volumes and fails closed without its map" begin
    b=ModelBuilder()
    points,tets=tet_box((.008,.006,.002),(2,2,2))
    part=add_deformable!(b,points,tets;place=p->p+PS.Vec3d(0,0,.1p[2]^2))
    piece=b.deformables[part]
    old=b.meshes[piece.mesh]
    proxy=PS.quadratic_surface_proxy(piece.x,PS.quadratic_boundary_faces(piece);
        initial=piece.initial,tolerance=1e-7)
    @test length(proxy.initial)>length(old.vertices)
    invalid=merge(proxy,(initial=proxy.initial .+ Ref(PS.Vec3d(1,0,0)),))
    @test_throws ArgumentError PS.install_surface_proxies!(b,Dict(part=>invalid))
    @test b.meshes[piece.mesh]===old
    reversed_faces=copy(proxy.faces)
    a,c,d=reversed_faces[1]
    reversed_faces[1]=(a,d,c)
    @test_throws ArgumentError PS.install_surface_proxies!(b,Dict(part=>merge(proxy,(faces=reversed_faces,))))
    @test b.meshes[piece.mesh]===old
    maps=PS.install_surface_proxies!(b,Dict(part=>proxy))
    @test b.deformables[part]===piece
    @test b.meshes[piece.mesh].static==old.static
    @test b.meshes[piece.mesh].vertices==proxy.initial
    @test_throws ArgumentError FEMSolver(Array,b)
    @test_throws ArgumentError Recording(b;compact=true)
    model=PS.MappedFEMModel(PS.FEMModel(Array,b),PS.FEMSurfaceTransfer(Array,b,maps))
    @test model isa PS.MappedFEMModel
    rec=Recording(b;compact=true,surface_maps=maps)
    @test rec isa PS.MappedRecording
    @test model.transfer.embedding.points==length(proxy.initial)
    @test_throws ArgumentError PS.install_surface_proxies!(b,Dict(part=>proxy);maximum_error=1e-9)
end
