using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:projection_random_cases) || include("ccd_projection.jl")
isdefined(@__MODULE__,:BACKEND) || error("select a coordinated GPU or explicit software backend")

function projection_axis_kernel!(out,scene,axes,radii,kind,scale::Float64)
    i=PS.KI.get_global_id(Int32).x
    i<=length(out) || return nothing
    ids=ntuple(k->Int32(4)*(i-Int32(1))+Int32(k),Val(4))
    @inbounds axis=PS.Vec3d(axes[1,i],axes[2,i],axes[3,i])
    @inbounds out[i]=UInt32(PS.ccd_projection_axis_separated(scene,ids,kind,radii[i],axis,scale))
    return nothing
end

function projection_prefix_kernel!(out,scene,kind=Val(false))
    PS.KI.get_global_id(Int32).x==1 || return nothing
    @inbounds out[1]=PS.filtered_swept_accumulate!(scene,Int32.((1,2,3,4)),kind,0f0,true)
    return nothing
end

function projection_moving_axis_kernel!(out,scene,axes,radii,kind)
    i=PS.KI.get_global_id(Int32).x
    i<=length(out) || return nothing
    ids=ntuple(k->Int32(4)*(i-Int32(1))+Int32(k),Val(4))
    @inbounds begin
        j=i==length(out) ? Int32(1) : i+Int32(1)
        a0=PS.Vec3d(axes[1,i],axes[2,i],axes[3,i])
        a1=PS.Vec3d(axes[1,j],axes[2,j],axes[3,j])
        out[i]=UInt32(PS.ccd_projection_moving_axis_separated(scene,ids,kind,radii[i],a0,a1))
    end
    return nothing
end

function projection_moving_fixture_kernel!(out,scene,ids,kind)
    PS.KI.get_global_id(Int32).x==1 || return nothing
    @inbounds out[1]=UInt32(PS.ccd_projected_moving_separated(scene,ids,kind,0f0,Val(true)))
    return nothing
end

@testset "device moving-axis certificate matches exact-qualified host" begin
    cases=projection_random_cases()
    scene=(;start=PS.upload(BACKEND,cases.start),finish=PS.upload(BACKEND,cases.finish))
    axes=PS.upload(BACKEND,reduce(hcat,cases.axes))
    radii=PS.upload(BACKEND,cases.radii)
    out=PS.upload(BACKEND,zeros(UInt32,length(cases.axes)))
    for kind in (Val(true),Val(false))
        expected=[UInt32(PS.ccd_projection_moving_axis_separated(cases,ntuple(k->Int32(4(i-1)+k),4),
            kind,cases.radii[i],cases.axes[i],cases.axes[mod1(i+1,length(cases.axes))])) for i in eachindex(cases.axes)]
        PS.launch!(BACKEND,projection_moving_axis_kernel!,(out,scene,axes,radii,kind),length(expected);group=64)
        actual=Array(out)
        @test actual==expected
        for i in findall(==(UInt32(1)),actual)
            @test rational_moving_projection_certificate(cases,ntuple(k->Int32(4(i-1)+k),4),
                kind,cases.radii[i],cases.axes[i],cases.axes[mod1(i+1,length(cases.axes))])
        end
    end
    fixture=captured_moving_projection_fixture()
    captured=(;start=PS.upload(BACKEND,fixture.start),finish=PS.upload(BACKEND,fixture.finish))
    flag=PS.upload(BACKEND,UInt32[0])
    for (ids,kind) in ((Int32.((2,4,5,6)),Val(true)),
            (Int32.((1,2,6,4)),Val(false)),(Int32.((2,3,6,4)),Val(false)))
        PS.launch!(BACKEND,projection_moving_fixture_kernel!,(flag,captured,ids,kind),1;group=64)
        @test only(Array(flag))==1
        copyto!(flag,UInt32[0])
        graph=PS.Mantle.Graph(PS.Mantle.todevice(BACKEND))
        PS.launch!(graph,projection_moving_fixture_kernel!,(flag,captured,ids,kind),1;group=64)
        plan=PS.Mantle.Plan(graph;policy=PS.Mantle.InOrder())
        PS.Mantle.record!(plan)
        for _ in 1:2
            PS.Mantle.run!(plan);PS.Mantle.waitfor!(plan)
            @test only(Array(flag))==1
        end
    end
end

@testset "device outward projection agrees with rational-qualified host cases" begin
    cases=projection_random_cases()
    normalized=[axis/maximum(abs,axis) for axis in cases.axes]
    scene=(;start=PS.upload(BACKEND,cases.start),finish=PS.upload(BACKEND,cases.finish))
    axes=PS.upload(BACKEND,reduce(hcat,normalized))
    radii=PS.upload(BACKEND,cases.radii)
    out=PS.upload(BACKEND,zeros(UInt32,length(cases.axes)))
    for kind in (Val(true),Val(false)),scale in (1.,.5,.125)
        expected=[UInt32(PS.ccd_projection_axis_separated(cases,ntuple(k->Int32(4(i-1)+k),4),
            kind,cases.radii[i],normalized[i],scale)) for i in eachindex(normalized)]
        PS.launch!(BACKEND,projection_axis_kernel!,(out,scene,axes,radii,kind,scale),length(expected);group=64)
        @test Array(out)==expected
        for i in findall(==(UInt32(1)),Array(out))
            @test rational_projection_certificate(cases,ntuple(k->Int32(4(i-1)+k),4),
                kind,cases.radii[i],normalized[i],scale)
        end
    end
    prefix=captured_projection_prefix_fixture()
    prefix_scene=(;start=PS.upload(BACKEND,prefix.start),finish=PS.upload(BACKEND,prefix.finish),
        refinement=Val(true),bound=PS.upload(BACKEND,prefix.bound),status=PS.upload(BACKEND,prefix.status))
    flag=PS.upload(BACKEND,UInt32[0])
    PS.launch!(BACKEND,projection_prefix_kernel!,(flag,prefix_scene),1;group=64)
    @test only(Array(flag))==0
    @test only(Array(prefix_scene.status))==0
    @test reinterpret(Float32,only(Array(prefix_scene.bound)))==1f0
    positive=captured_projection_positive_fixture()
    positive_scene=(;start=PS.upload(BACKEND,positive.start),finish=PS.upload(BACKEND,positive.finish),
        refinement=Val(true),bound=PS.upload(BACKEND,positive.bound),status=PS.upload(BACKEND,positive.status))
    PS.launch!(BACKEND,projection_prefix_kernel!,(flag,positive_scene,Val(true)),1;group=64)
    @test only(Array(flag))==1
    @test only(Array(positive_scene.status))==1
    @test reinterpret(Float32,only(Array(positive_scene.bound)))==.9f0/128
    b=ModelBuilder()
    for z in (0.,.01)
        add_sheet!(b,SheetGrid(.01,.01,2,2),SheetMaterial();place=p->PS.Vec3d(p[1],p[2],z))
    end
    ccd=PS.SweptCollision(BACKEND,b)
    fixture=captured_projection_fixture()
    start=Array(ccd.start);finish=Array(ccd.finish)
    start[1:5]=fixture.start;finish[1:5]=fixture.finish
    copyto!(ccd.start,start);copyto!(ccd.finish,finish)
    faces=Array(ccd.faces);faces[1]=(1,2,3);faces[2]=(4,3,5)
    copyto!(ccd.faces,faces)
    owners=Array(ccd.owner);owners[1]=1;owners[2]=2
    copyto!(ccd.owner,owners);copyto!(ccd.radius,zeros(Float32,length(ccd.radius)))
    work=(;pairs=PS.upload(BACKEND,[(Int32(1),Int32(2))]),flags=PS.upload(BACKEND,UInt32[123]))
    one=PS.upload(BACKEND,Int32[1])
    PS.launch!(BACKEND,PS.ccd_queue_checks!,(work,ccd,one,one,Int32(1),true),1;group=64)
    @test only(Array(work.flags))==0
    @test only(Array(ccd.status))==0
    @test reinterpret(Float32,only(Array(ccd.bound)))==1f0
end
