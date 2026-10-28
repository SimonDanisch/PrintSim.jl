# RayMakie already accepts device Point3f positions and device triangle indices.
# Keep display reconstruction there too: upload recorded DOFs, not a rebuilt
# multi-million-vertex CPU mesh for every frame. Physics is never rerun here.
struct RecordingSurface{B,P,F,G,D}
    backend::B
    points::P
    faces::F
    geometry::G
    dofs::D
end

function RecordingSurface(backend,rec::Union{Recording,MappedRecording},frame::DynamicsFrame)
    faces=upload(backend,rec.faces)
    if !rec.compact
        return RecordingSurface(backend,upload(backend,frame.surface),faces,nothing,nothing)
    end
    count=rec.sheet_vertices+length(rec.rest_surface)
    owner=zeros(Int32,length(rec.rest_surface))
    for (mesh,range) in enumerate(rec.mesh_vertices),i in range
        owner[i-rec.sheet_vertices]=rec.mesh_body[mesh]
    end
    source=rec isa MappedRecording ? rec.transfer.embedding : nothing
    embedding=source===nothing ? nothing :
        (;node=upload(backend,source.node),weight=upload(backend,source.weight))
    rows=source===nothing ? zeros(Int32,length(owner)) : rec.transfer.point_row
    geometry=(;sheets=rec.sheet_vertices,body=upload(backend,owner),
        rest=upload(backend,rec.rest_surface),coordinates=upload(backend,rec.coordinates),
        fem_node=upload(backend,rec.fem_node),point_row=upload(backend,rows),embedding)
    dofs=(;sheets=KI.zeros(backend,Position,max(1,length(frame.sheets))),
        bodies=KI.zeros(backend,Position,max(1,length(frame.bodies))),
        deformables=KI.zeros(backend,Position,max(1,length(frame.deformables))),
        one=upload(backend,Float32[1]))
    result=RecordingSurface(backend,KI.zeros(backend,Point3f,count),faces,geometry,dofs)
    update_recording_surface!(result,frame)
    return result
end

function recorded_surface_points!(points,geometry,dofs,first::Int32,count::Int32)
    lane=KI.get_global_id(Int32).x
    lane<=count || return nothing
    i=first+lane-Int32(1)
    @inbounds begin
        if i<=geometry.sheets
            points[i]=Point3f(approximate(dofs.sheets[i]))
            return nothing
        end
        point=i-Int32(geometry.sheets)
        row=geometry.point_row[point]
        node=geometry.fem_node[point]
        body=geometry.body[point]
        one=dofs.one[1]
        if geometry.embedding!==nothing && row!=0
            p=surface_map_position(geometry.embedding,row,dofs.deformables,one)
            points[i]=Point3f(approximate(p))
        elseif node!=0
            points[i]=Point3f(approximate(dofs.deformables[node]))
        elseif body!=0
            p=affine_point(dofs.bodies,body,geometry.coordinates[point],one)
            points[i]=Point3f(approximate(p))
        else
            points[i]=geometry.rest[point]
        end
    end
    return nothing
end

function update_recording_surface!(surface::RecordingSurface,frame::DynamicsFrame)
    if surface.geometry===nothing
        length(frame.surface)==length(surface.points) || throw(DimensionMismatch("recording surface changed shape"))
        copyto!(surface.points,frame.surface)
    else
        for name in (:sheets,:bodies,:deformables)
            host=getfield(frame,name)
            destination=getfield(surface.dofs,name)
            max(1,length(host))==length(destination) || throw(DimensionMismatch("recorded DOF count changed"))
            isempty(host) || copyto!(destination,host)
        end
        launch_indexed_batches!(surface.backend,recorded_surface_points!,
            (surface.points,surface.geometry,surface.dofs),length(surface.points);batch=65536)
    end
    Mantle.waitidle(Mantle.todevice(surface.backend))
    return surface.points
end
