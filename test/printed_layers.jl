using PrintSim,Test
import PrintSim as PS

@testset "bore survives an untagged exterior bridge perimeter" begin
    function ring(radius,outer)
        points=[PS.Vec2d(radius*cos(t),radius*sin(t)) for t in range(0,2pi;length=97)[1:end-1]]
        push!(points,first(points))
        PS.BeadPath(points,.2,.2,.4,outer)
    end
    for island in (false,true)
        tagged=[ring(5.,true),ring(2.,true)]
        untagged=[ring(5.,false),ring(2.,true)]
        if island
            push!(tagged,ring(.5,true));push!(untagged,ring(.5,true))
        end
        expected=PS.layer_region(tagged)
        actual=PS.layer_region(untagged)
        @test PS.GEOS.area(PS.GEOS.difference(actual,expected))+
            PS.GEOS.area(PS.GEOS.difference(expected,actual))<1e-9
        @test !PS.GEOS.covers(actual,PS.GEOS.Point(1.,0.))
        @test PS.GEOS.covers(actual,PS.GEOS.Point(4.,0.))
        @test PS.GEOS.covers(actual,PS.GEOS.Point(0.,0.))==island
    end
end
