using PrintSim,Test,Random
import PrintSim as PS

function captured_projection_fixture()
    start=[
        PS.Position(PS.Vec3f(-.1268,-.024722679,.071133584),PS.Vec3f(5.667572f-9,7.3287584f-11,1.1776163f-9)),
        PS.Position(PS.Vec3f(-.1268,-.024722718,.071133554),PS.Vec3f(5.667572f-9,-6.1825667f-10,-4.7377213f-10)),
        PS.Position(PS.Vec3f(-.1268,-.024722686,.0711336),PS.Vec3f(5.667572f-9,-6.173648f-10,-3.2881915f-9)),
        PS.Position(PS.Vec3f(-.1268,-.024712313,.071141794),PS.Vec3f(5.667572f-9,-9.3107f-11,-2.2992384f-9)),
        PS.Position(PS.Vec3f(-.1268,-.024722695,.071133606),PS.Vec3f(5.667572f-9,5.5462795f-10,-3.03419f-10))]
    finish=[
        PS.Position(PS.Vec3f(-.1268017,-.024725894,.07114215),PS.Vec3f(3.7242565f-10,-2.8899508f-10,1.2361651f-9)),
        PS.Position(PS.Vec3f(-.1268017,-.024725934,.07114212),PS.Vec3f(3.733404f-10,8.9391433f-10,-3.227645f-10)),
        PS.Position(PS.Vec3f(-.1268017,-.024725903,.07114217),PS.Vec3f(4.1213974f-10,8.7803603f-10,-3.1842158f-9)),
        PS.Position(PS.Vec3f(-.1268017,-.024715533,.07115033),PS.Vec3f(1.5100057f-9,8.335705f-10,2.5377875f-9)),
        PS.Position(PS.Vec3f(-.1268017,-.02472591,.071142174),PS.Vec3f(4.1531445f-10,1.621085f-10,-1.8096696f-10))]
    return (;start,finish,refinement=Val(true),bound=UInt32[reinterpret(UInt32,1f0)],status=UInt32[0])
end


function captured_projection_prefix_fixture()
    start=[
        PS.Position(PS.Vec3f(-.1268,-.024691701,.07115806),PS.Vec3f(5.667572f-9,-8.6564106f-10,1.4479508f-9)),
        PS.Position(PS.Vec3f(-.1268,-.024681473,.07116614),PS.Vec3f(5.667572f-9,8.72205f-10,-2.5355884f-10)),
        PS.Position(PS.Vec3f(-.1268,-.024681464,.071166135),PS.Vec3f(5.667572f-9,-4.1287457f-10,-3.0940173f-9)),
        PS.Position(PS.Vec3f(-.1268,-.024691693,.07115805),PS.Vec3f(5.667572f-9,-2.8807545f-10,-1.3923491f-9))]
    finish=[
        PS.Position(PS.Vec3f(-.1268017,-.024694929,.07116655),PS.Vec3f(4.0444044f-9,-4.810465f-10,1.3779732f-9)),
        PS.Position(PS.Vec3f(-.1268017,-.024684703,.071174614),PS.Vec3f(5.264738f-9,-7.556588f-10,-2.631407f-9)),
        PS.Position(PS.Vec3f(-.1268017,-.024684696,.0711746),PS.Vec3f(5.251835f-9,-1.5412506f-10,1.9627955f-9)),
        PS.Position(PS.Vec3f(-.1268017,-.024694921,.071166545),PS.Vec3f(4.0389567f-9,1.00356876f-10,-1.4905118f-9))]
    return (;start,finish,refinement=Val(true),bound=UInt32[reinterpret(UInt32,1f0)],status=UInt32[0])
end

function captured_projection_positive_fixture()
    start=[
        PS.Position(PS.Vec3f(.119,-.020580754,.06591303),PS.Vec3f(-2.7418137f-9,-6.80428f-10,5.8870064f-10)),
        PS.Position(PS.Vec3f(.118999995,-.020586023,.065783),PS.Vec3f(-2.9123304f-10,-1.9376975f-10,1.3459843f-9)),
        PS.Position(PS.Vec3f(.118999995,-.020616472,.06628803),PS.Vec3f(-2.9123304f-10,2.0325844f-10,4.6455784f-10)),
        PS.Position(PS.Vec3f(.118999995,-.020524466,.06620596),PS.Vec3f(-2.9123304f-10,5.2516f-11,-3.373246f-10))]
    finish=[start[1],
        PS.Position(PS.Vec3f(.11900049,-.02058634,.06578487),PS.Vec3f(-5.5715443f-10,-2.1876584f-10,-3.6403003f-9)),
        PS.Position(PS.Vec3f(.1190005,-.020616941,.06628994),PS.Vec3f(2.0911974f-9,3.8281756f-10,2.9303482f-9)),
        PS.Position(PS.Vec3f(.11900048,-.020524912,.06620774),PS.Vec3f(1.881789f-9,8.7736474f-10,-9.956395f-10))]
    return (;start,finish,refinement=Val(true),bound=UInt32[reinterpret(UInt32,1f0)],status=UInt32[0])
end

# Exact rational oracle for the certificate's finite computed axis. The axis
# need not approximate the true normal accurately; any fixed axis is valid.
function rational_projection_certificate(ccd,ids,::Val{VF},radius,axis,prefix_scale=1.) where VF
    n=axis/maximum(abs,axis)
    rational(v)=Rational{BigInt}(v)
    coordinate(p,k)=rational(p.hi[k])+rational(p.lo[k])
    dotdiff(a,b,t)=sum(rational(n[k])*((1-t)*(coordinate(ccd.start[a],k)-coordinate(ccd.start[b],k))+
        t*(coordinate(ccd.finish[a],k)-coordinate(ccd.finish[b],k))) for k in 1:3)
    values=Rational{BigInt}[]
    for time in (zero(Rational{BigInt}),rational(prefix_scale))
        if VF
            append!(values,[dotdiff(ids[1],ids[b],time) for b in 2:4])
        else
            append!(values,[dotdiff(ids[a],ids[b],time) for a in 1:2 for b in 3:4])
        end
    end
    margin=rational(radius)*sum(abs,rational.(n))
    return minimum(values)>margin || maximum(values)<-margin
end

function projection_random_cases(n=128)
    rng=MersenneTwister(0x51ccdc)
    starts=PS.Position[];finishes=PS.Position[];axes=PS.Vec3d[];radii=Float32[]
    for _ in 1:n
        origin=PS.Vec3f(randn(rng,3).*10.0^rand(rng,-8:8))
        scale=Float32(10.0^rand(rng,-12:2))
        for _ in 1:4
            p=PS.Position(origin,PS.Vec3f(randn(rng,3).*scale))
            push!(starts,p)
            push!(finishes,PS.Position(p.hi,p.lo+PS.Vec3f(randn(rng,3).*scale)))
        end
        push!(axes,PS.Vec3d(randn(rng,3)))
        push!(radii,rand(rng,Bool) ? 0f0 : .01f0*scale)
    end
    return (;start=starts,finish=finishes,axes,radii)
end

@testset "outward projected trajectory certificates preserve exact stored positions" begin
    fixture=captured_projection_fixture()
    ids=Int32.((1,4,3,5))
    # Pin the exact geometry, not a false positive from an older unfiltered
    # solver. A fixed axis from the captured triangle edge separates the
    # exact stored endpoints for the complete trajectory.
    position(p)=PS.Vec3d(p.hi)+PS.Vec3d(p.lo)
    edge=position(fixture.start[ids[4]])-position(fixture.start[ids[3]])
    @test rational_projection_certificate(fixture,ids,Val(true),0f0,PS.Vec3d(0,edge[3],-edge[2]))
    @test PS.ccd_projected_separated(fixture,ids,Val(true),0f0,Val(true))
    @test !PS.ccd_projected_separated(fixture,ids,Val(true),0f0,Val(false))
    @test !PS.ccd_projected_separated(fixture,ids,Val(true),1f-4,Val(true))
    @test PS.filtered_swept_triangle_pair!(fixture,Int32.((1,2,3)),Int32.((4,3,5)),0f0,true)==0
    @test only(fixture.bound)==reinterpret(UInt32,1f0)
    prefix_fixture=captured_projection_prefix_fixture()
    prefix_ids=Int32.((1,2,3,4))
    @test !PS.ccd_projected_separated(prefix_fixture,prefix_ids,Val(false),0f0,Val(true))
    prefix=PS.ccd_projected_prefix(prefix_fixture,prefix_ids,Val(false),0f0,Val(true))
    @test prefix==.5f0
    @test PS.filtered_swept_accumulate!(prefix_fixture,prefix_ids,Val(false),0f0,true)==UInt32(0)
    @test only(prefix_fixture.status)==UInt32(0)
    @test reinterpret(Float32,only(prefix_fixture.bound))==1f0
    positive=captured_projection_positive_fixture()
    reference=PS.swept_primitive(positive,prefix_ids,Val(true),0f0,true)
    @test reference.hit && reference.time>0 && reference.resolved
    @test PS.filtered_swept_accumulate!(positive,prefix_ids,Val(true),0f0,true)==0
    cap=reinterpret(Float32,only(positive.bound))
    # Preserve at least the certified dyadic-prefix progress. The current
    # solver can advance further; certify that returned step independently.
    @test cap>=.9f0/128
    @test rational_projection_certificate(positive,prefix_ids,Val(true),0f0,PS.Vec3d(1,0,0),Float64(cap))
    cases=projection_random_cases()
    certified=0
    for kind in (Val(true),Val(false)),case in eachindex(cases.axes),scale in (1.,.5,.125)
        indices=ntuple(k->Int32(4(case-1)+k),4)
        result=PS.ccd_projection_axis_separated(cases,indices,kind,cases.radii[case],cases.axes[case],scale)
        if result
            certified+=1
            @test rational_projection_certificate(cases,indices,kind,cases.radii[case],cases.axes[case],scale)
        end
    end
    @test certified>0
    @test PS.ccd_projection_lower(0.)<0<PS.ccd_projection_upper(0.)
    @test isnan(PS.ccd_projection_wide(reinterpret(Float32,UInt32(1))))
    # A trailing 1e-30 is real even when Float64(hi)+Float64(lo) rounds to hi.
    p=PS.Position(PS.Vec3f(1,0,0),PS.Vec3f(1f-30,0,0))
    q=PS.Position(PS.Vec3f(1,0,0))
    lo,hi=PS.ccd_projection_difference(p,q,1)
    @test 0<lo<=Float64(p.lo[1])<=hi
    touching=(;start=fill(q,4),finish=fill(q,4))
    @test !PS.ccd_projection_axis_separated(touching,Int32.((1,2,3,4)),Val(true),0f0,PS.Vec3d(1,0,0))
    @test PS.ccd_projected_prefix(merge(touching,(;refinement=Val(true))),Int32.((1,2,3,4)),Val(true),0f0,Val(true))==0f0
    @test !PS.ccd_projection_axis_separated(fixture,ids,Val(true),0f0,PS.Vec3d(0,1,0),.3)
    for radius in (1f-9,1f-30),delta in (prevfloat(radius),radius,nextfloat(radius))
        point=PS.Position(q.hi,PS.Vec3f(delta,0,0))
        scene=(;start=[point,q,q,q],finish=[point,q,q,q])
        separated=PS.ccd_projection_axis_separated(scene,Int32.((1,2,3,4)),Val(true),radius,PS.Vec3d(1,0,0))
        @test separated==(delta>radius)
    end
    invalid=(;start=[PS.Position(PS.Vec3f(NaN,0,0)),q,q,q],finish=fill(q,4))
    @test !PS.ccd_projection_axis_separated(invalid,Int32.((1,2,3,4)),Val(true),0f0,PS.Vec3d(0,1,0))
end

include("ccd_projection_moving.jl")
