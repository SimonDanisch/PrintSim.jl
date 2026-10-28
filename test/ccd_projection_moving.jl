# Exact endpoints of the full-scene CFL trial's limiting pair: two proxy facets
# on quadratic parent 38386 of idler_spring_r. Input SHA f224af28adeaa3706b8023f6...
function captured_moving_projection_fixture()
    start=[
        PS.Position(PS.Vec3f(.12,-.022626199,.05950149),PS.Vec3f(-2.317791f-9,3.193446f-10,-1.8191464f-9)),
        PS.Position(PS.Vec3f(.12,-.0225916,.059502136),PS.Vec3f(-2.317791f-9,-1.3982593f-10,-6.4538996f-10)),
        PS.Position(PS.Vec3f(.12,-.0225949,.059502073),PS.Vec3f(-2.317791f-9,7.980521f-10,9.167466f-10)),
        PS.Position(PS.Vec3f(.12,-.022613,.059501737),PS.Vec3f(-2.317791f-9,2.9312255f-10,5.491596f-11)),
        PS.Position(PS.Vec3f(.12,-.022530604,.05950327),PS.Vec3f(-2.317791f-9,-6.514404f-10,1.8223345f-9)),
        PS.Position(PS.Vec3f(.12,-.022557002,.05950278),PS.Vec3f(-2.317791f-9,-5.989964f-10,-1.7847599f-9))]
    finish=[
        PS.Position(PS.Vec3f(.12,-.022626188,.059501514),PS.Vec3f(1.5972789f-9,-8.748118f-10,-9.447867f-10)),
        PS.Position(PS.Vec3f(.12,-.02259159,.05950216),PS.Vec3f(1.5520035f-9,5.301617f-10,-6.9155126f-11)),
        PS.Position(PS.Vec3f(.12,-.02259489,.059502095),PS.Vec3f(1.5563149f-9,-3.947792f-10,1.5213671f-9)),
        PS.Position(PS.Vec3f(.12,-.022612989,.05950176),PS.Vec3f(1.5800385f-9,-9.003204f-10,8.1537893f-10)),
        PS.Position(PS.Vec3f(.12,-.022530595,.059503295),PS.Vec3f(1.4722494f-9,2.1349533f-11,-1.8493838f-9)),
        PS.Position(PS.Vec3f(.12,-.022556992,.059502803),PS.Vec3f(1.5067332f-9,7.24493f-11,-1.5054318f-9))]
    return (;start,finish)
end

function rational_moving_projection_certificate(ccd,ids,::Val{VF},radius,axis0,axis1) where VF
    magnitude=max(maximum(abs,axis0),maximum(abs,axis1))
    n0,n1=axis0/magnitude,axis1/magnitude
    Q=Rational{BigInt}
    coordinate(p,k)=Q(p.hi[k])+Q(p.lo[k])
    dotdiff(endpoint,a,b,normal)=sum(Q(normal[k])*(coordinate(endpoint[ids[a]],k)-coordinate(endpoint[ids[b]],k)) for k in 1:3)
    pairs=VF ? [(1,b) for b in 2:4] : [(a,b) for a in 1:2 for b in 3:4]
    coefficients=Q[]
    for (a,b) in pairs
        append!(coefficients,(dotdiff(ccd.start,a,b,n0),
            (dotdiff(ccd.start,a,b,n1)+dotdiff(ccd.finish,a,b,n0))/2,
            dotdiff(ccd.finish,a,b,n1)))
    end
    threshold=Q(radius)*max(sum(Q(abs(v)) for v in n0),sum(Q(abs(v)) for v in n1))
    return minimum(coefficients)>threshold || maximum(coefficients)<-threshold
end

@testset "moving-axis Bernstein hull is outward and conservative" begin
    cases=projection_random_cases()
    certified=0
    for kind in (Val(true),Val(false)),i in eachindex(cases.axes)
        ids=ntuple(k->Int32(4(i-1)+k),4)
        axis0=cases.axes[i]
        axis1=cases.axes[mod1(i+1,length(cases.axes))]
        result=PS.ccd_projection_moving_axis_separated(cases,ids,kind,cases.radii[i],axis0,axis1)
        if result
            certified+=1
            @test rational_moving_projection_certificate(cases,ids,kind,cases.radii[i],axis0,axis1)
        end
    end
    @test certified>0
    fixture=captured_moving_projection_fixture()
    for (ids,kind) in ((Int32.((2,4,5,6)),Val(true)),
            (Int32.((1,2,6,4)),Val(false)),(Int32.((2,3,6,4)),Val(false)))
        @test !PS.ccd_projected_separated(fixture,ids,kind,0f0,Val(true))
        @test PS.ccd_projected_moving_separated(fixture,ids,kind,0f0,Val(true))
        @test !PS.ccd_projected_moving_separated(fixture,ids,kind,0f0,Val(false))
    end
    pos(v)=PS.Position(PS.Vec3f(v...))
    crossing=(;start=pos.([(0,0,1),(-1,-1,0),(1,-1,0),(0,1,0)]),
        finish=pos.([(0,0,-1),(-1,-1,0),(1,-1,0),(0,1,0)]))
    ids=Int32.((1,2,3,4))
    @test !PS.ccd_projected_moving_separated(crossing,ids,Val(true),0f0,Val(true))
    @test !PS.ccd_projection_moving_axis_separated(crossing,ids,Val(true),0f0,PS.Vec3d(NaN,0,0),PS.Vec3d(1,0,0))
    q=pos((1,0,0))
    touching=(;start=fill(q,4),finish=fill(q,4))
    @test !PS.ccd_projected_moving_separated(touching,ids,Val(true),0f0,Val(true))
    @test !PS.ccd_projection_moving_axis_separated(touching,ids,Val(true),reinterpret(Float32,UInt32(1)),PS.Vec3d(1,0,0),PS.Vec3d(1,0,0))
    for radius in (1f-9,1f-30),delta in (prevfloat(radius),radius,nextfloat(radius))
        point=PS.Position(q.hi,PS.Vec3f(delta,0,0))
        scene=(;start=[point,q,q,q],finish=[point,q,q,q])
        @test PS.ccd_projection_moving_axis_separated(scene,ids,Val(true),radius,PS.Vec3d(1,0,0),PS.Vec3d(1,0,0))==(delta>radius)
    end
end
