using PrintSim,Test
import PrintSim as PS

function precision_ccd_fixture(;crossing=false)
    o=PS.Vec3d(.1,.05,-.02)
    a=(PS.Vec3d(0,0,0),PS.Vec3d(.001,0,0),PS.Vec3d(0,0,1e-9),PS.Vec3d(.001,0,1e-9))
    b=crossing ? (a[1],a[2],a[3]-PS.Vec3d(0,0,2e-9),a[4]-PS.Vec3d(0,0,2e-9)) : a
    return (start=PS.Position.([o+p for p in a]),finish=PS.Position.([o+p for p in b]),refinement=Val(true))
end

# Captured from the full installed idler-spring surface. Nonincident triangles
# have a certified ~20 nm normal gap but exhausted spatial inclusion's queue.
function spatial_ccd_fixture(; thin=false)
    points = thin ? PS.Vec3d[
        (-.11679999499999999,-.017557492306453856,.06568407214124594),
        (-.11669999500000006,-.017557492306453856,.06568407214124594),
        (-.11669999500000006,-.01755725623945234,.06568391484198444),
        (-.11669999500000006,-.017496482430449767,.06564341565781362),
        (-.11679999499999999,-.017557423124391264,.06568402604295764),
        (-.11679999499999999,-.017557457715422553,.0656840490921029)
    ] : PS.Vec3d[
        (-.1182999950000001, -.020533456373521997, .06529354242848423),
        (-.1182999950000001, -.020571058709563314, .06526108672694242),
        (-.118199995, -.020571058709563314, .06526108672694242),
        (-.11809999500000001, -.020514684474666034, .0653098651323436),
        (-.118199995, -.020571072191165496, .06526110238764504),
        (-.118199995, -.02057114334383059, .0652610408526455)]
    positions = PS.Position.(points)
    return (; start=positions, finish=positions, refinement=Val(true),
        bound=UInt32[reinterpret(UInt32,1f0)], status=UInt32[0])
end

@testset "captured idler surface gap is certified without exclusions" begin
    for thin in (false,true)
        ccd=spatial_ccd_fixture(;thin)
        @test PS.swept_triangle_pair!(ccd,Int32.((1,2,3)),Int32.((4,5,6)),0f0,true)==0
        @test only(ccd.status)==0
    end
end

@testset "wide inclusion refinement resolves nanometre gaps without losing crossings" begin
    ids=Int32.((1,2,3,4))
    separated=precision_ccd_fixture()
    coarse=PS.swept_primitive(merge(separated,(refinement=Val(false),)),ids,Val(false),0f0,true)
    @test coarse.hit && coarse.time==0
    precise=PS.swept_primitive(separated,ids,Val(false),0f0,true)
    @test !precise.hit && precise.resolved
    crossing=PS.swept_primitive(precision_ccd_fixture(;crossing=true),ids,Val(false),0f0,true)
    @test crossing.hit && 0<crossing.time<=.5f0
    thick=PS.swept_primitive(separated,ids,Val(false),1f-4,true)
    @test thick.hit && thick.time==0
end
