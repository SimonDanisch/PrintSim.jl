# Narrow-phase per-lane receipt: distinguishes dispatch/ID errors from arithmetic
# and from cross-dispatch aggregation. Requires explicit BACKEND from caller.
using PrintSim, Test
import PrintSim as PS
isdefined(@__MODULE__,:captured_projection_positive_fixture) || include("ccd_projection.jl")

function primitive_receipt!(ids_out,result_out,scene,faces,kind)
    lane=PS.KI.get_global_id(Int32).x
    lane<=(kind isa Val{true} ? 6 : 9) || return nothing
    @inbounds begin
        a,b=faces[1],faces[2]
        if kind isa Val{true}
            if lane<=3
                ids=(a[lane],b...)
            else
                ids=(b[lane-Int32(3)],a...)
            end
            incident=ids[1] in (ids[2],ids[3],ids[4])
        else
            edge=lane-Int32(1)
            i,j=edge÷Int32(3)+Int32(1),edge%Int32(3)+Int32(1)
            ids=(a[i],a[i==3 ? Int32(1) : i+Int32(1)],
                b[j],b[j==3 ? Int32(1) : j+Int32(1)])
            incident=ids[1]==ids[3] || ids[1]==ids[4] || ids[2]==ids[3] || ids[2]==ids[4]
        end
        ids_out[lane]=ids
        separated=PS.ccd_projected_separated(scene,ids,kind,0f0,scene.refinement)
        r=PS.swept_primitive(scene,ids,kind,0f0,true)
        prefix=PS.ccd_projected_prefix(scene,ids,kind,0f0,scene.refinement)
        result_out[lane]=(Float32(incident),Float32(separated),Float32(r.hit),r.time,Float32(r.resolved),prefix)
    end
    return nothing
end

function ccd_primitive_receipts(backend,fixture,face_ids)
    scene=(;start=PS.upload(backend,fixture.start),finish=PS.upload(backend,fixture.finish),refinement=Val(true))
    faces=PS.upload(backend,[Int32.(f) for f in face_ids])
    for kind in (Val(true),Val(false))
        n=kind isa Val{true} ? 6 : 9
        ids=PS.upload(backend,fill(ntuple(_->Int32(0),4),n))
        result=PS.upload(backend,fill(ntuple(_->0f0,6),n))
        PS.launch!(backend,primitive_receipt!,(ids,result,scene,faces,kind),n;group=64)
        actual_ids,actual=Array(ids),Array(result)
        println((kind,actual_ids,actual));flush(stdout)
        for k in 1:n
            id=actual_ids[k]
            a,b=Int32.(face_ids[1]),Int32.(face_ids[2])
            expected=if kind isa Val{true}
                k<=3 ? (a[k],b...) : (b[k-3],a...)
            else
                i,j=(k-1)÷3+1,(k-1)%3+1
                (a[i],a[mod1(i+1,3)],b[j],b[mod1(j+1,3)])
            end
            @test id==expected
            id==expected || continue # never dereference a corrupted device ID
            separated=PS.ccd_projected_separated(fixture,id,kind,0f0,Val(true))
            host=(;start=fixture.start,finish=fixture.finish,refinement=Val(true))
            r=PS.swept_primitive(host,id,kind,0f0,true)
            prefix=PS.ccd_projected_prefix(host,id,kind,0f0,Val(true))
            println((kind,k,id,device=actual[k],host=(separated,r,prefix)))
        end
    end
    return nothing
end
