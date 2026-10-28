using PrintSim,Test
import PrintSim as PS

function captured_initial_crossing_fixture()
    return [
        PS.Position(PS.Vec3f(.118499994,-.017581606,.070177235),PS.Vec3f(7.2204587f-10,-3.2873682f-10,-2.4301272f-9)),
        PS.Position(PS.Vec3f(.11839999,-.017581606,.070177235),PS.Vec3f(2.414818f-9,-3.2873682f-10,-2.4301272f-9)),
        PS.Position(PS.Vec3f(.11839999,-.017603561,.070187934),PS.Vec3f(2.414818f-9,1.6170668f-10,-1.90709f-9)),
        PS.Position(PS.Vec3f(.1184,-.017511696,.07013319),PS.Vec3f(-3.5762726f-11,-8.4330154f-10,-2.9317482f-9)),
        PS.Position(PS.Vec3f(.1184,-.017746044,.07027279),PS.Vec3f(-3.5762726f-11,3.520415f-10,-1.5631654f-9)),
        PS.Position(PS.Vec3f(.1184,-.017510723,.07013897),PS.Vec3f(-3.5762726f-11,2.2554847f-10,-1.5693387f-10))]
end

function initial_crossing_cases()
    synthetic=PS.Position.(PS.Vec3d[(-1,0,0),(1,0,0),(0,2,0),
        (0,.5,-1),(0,1,1),(0,0,1)])
    return (synthetic,captured_initial_crossing_fixture())
end

@testset "initial surface crossings are not motion VF/EE coincidence queries" begin
    for points in initial_crossing_cases()
        a,b=Int32.((1,2,3)),Int32.((4,5,6))
        scene=(;start=points,finish=points,refinement=Val(true),
            bound=UInt32[reinterpret(UInt32,1f0)],status=UInt32[0])
        # This is NOT a motion-CCD false negative: its disjoint-start
        # precondition is violated. Initial geometry needs an additional test.
        for k in 1:3
            @test !PS.swept_primitive(scene,(a[k],b...),Val(true),0f0,true).hit
            @test !PS.swept_primitive(scene,(b[k],a...),Val(true),0f0,true).hit
        end
        for i in 1:3,j in 1:3
            @test !PS.swept_primitive(scene,(a[i],a[mod1(i+1,3)],b[j],b[mod1(j+1,3)]),Val(false),0f0,true).hit
        end
        @test PS.swept_triangle_pair!(scene,a,b,0f0,true) & UInt32(2)!=0
        @test only(scene.bound)==0
        @test only(scene.status) & UInt32(2)!=0
    end
end
