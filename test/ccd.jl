using Test, StaticArrays, LinearAlgebra
isdefined(@__MODULE__, :tight_ccd) || include(joinpath(@__DIR__, "..", "src", "dynamics", "ccd.jl"))

@testset "inclusion CCD detects crossings between separated endpoints" begin
    for T in (Float32, Float64)
        v(x, y, z) = SVector{3, T}(x, y, z)
        a, b, c = v(-1, -1, 0), v(1, -1, 0), v(0, 1, 0)
        query(x, y, vf; n = 4096) = tight_ccd(x, y, Val(vf), T(0), T(1e-4), Int32(n))
        hit = query((v(0, 0, 1), a, b, c), (v(0, 0, -1), a, b, c), true)
        @test hit.hit && hit.resolved
        @test 0 <= hit.time <= 0.5
        miss = query((v(3, 0, 1), a, b, c), (v(3, 0, -1), a, b, c), true)
        @test !miss.hit && miss.resolved && miss.time == 1
        edges0 = (v(-1, 0, 0), v(1, 0, 0), v(0, -1, 1), v(0, 1, 1))
        edges1 = (edges0[1], edges0[2], v(0, -1, -1), v(0, 1, -1))
        edgehit = query(edges0, edges1, false)
        @test edgehit.hit && edgehit.resolved && 0 <= edgehit.time <= 0.5
        # Degenerate edges are also point trajectories, including grazing.
        p, q = v(0, 0, 0), v(0, 1, 0)
        @test !query((p, p, q, q), (p, p, q, q), false).hit
        capped = query(edges0, edges1, false; n = 1)
        @test capped.hit && !capped.resolved && capped.time <= 0.5
        nan = v(NaN, 0, 0)
        @test !query((nan, p, q, q), (p, p, q, q), false).resolved
        # Finite inputs whose intermediate differences overflow cannot be
        # mistaken for a proven separation.
        big = v(floatmax(T), 0, 0)
        overflow = query((big, big, -big, -big), (-big, -big, big, big), false)
        @test overflow.hit && !overflow.resolved
        # Nonzero thickness must stop earlier than the zero-thickness impact.
        thick = tight_ccd(edges0, edges1, Val(false), T(0.1), T(1e-4), Int32(4096))
        @test thick.hit && thick.time <= T(0.45)
        # Parallel overlapping edge parameters must not exhaust the queue
        # before establishing any safe time for well-separated endpoints.
        parallel0=(v(0,0,0),v(.01,0,0),v(0,0,.001),v(.01,0,.001))
        parallel1=(parallel0[1],parallel0[2],v(0,0,-.001),v(.01,0,-.001))
        parallel=tight_ccd(parallel0,parallel1,Val(false),T(.0001),T(1e-8),Int32(512))
        @test parallel.hit && 0 < parallel.time <= T(.45)
    end
end

@testset "filtered planar certificates never turn uncertainty into separation" begin
    for k in 1:512
        scale = exp2(mod(k * 17, 1001) - 500)
        p = (sin(k) * scale, cos(k) * scale)
        q = (sin(2k) * scale, cos(3k) * scale)
        r = (sin(5k) * scale, cos(7k) * scale)
        sign = ccd_orient2_filter(p, q, r)
        a, b, c = map(x -> Rational{BigInt}.(x), (p, q, r))
        exact = Base.sign((a[1] - c[1]) * (b[2] - c[2]) - (a[2] - c[2]) * (b[1] - c[1]))
        @test sign == 2 || sign == exact
    end
    v(x, y) = SVector{3, Float64}(x, y, 0)
    edges = (v(0, 0), v(.01, .01), v(0, 1e-9), v(.01, .01 + 1e-9))
    separated = tight_ccd(edges, edges, Val(false), 0., 1e-12, Int32(8192))
    @test !separated.hit && separated.resolved
    crossing = (edges[1], edges[2], v(0, 1e-9), v(.01, .01 - 1e-9))
    @test tight_ccd(crossing, crossing, Val(false), 0., 1e-12, Int32(8192)).hit
    touching = (edges[1], edges[2], edges[1], v(0, .01))
    @test tight_ccd(touching, touching, Val(false), 0., 1e-12, Int32(8192)).hit
    @test ccd_orient2_filter((0., 0.), (1., 1.), (2., 2.)) == 2
    oblique(x,y)=SVector{3,Float64}(x+.3y,.4x+y,.7x+.9y)
    rotated=(oblique(0,0),oblique(.01,.01),oblique(0,1e-9),oblique(.01,.01+1e-9))
    @test ccd_planar_static_separated(rotated,rotated,Val(false),1e-16)
    @test !tight_ccd(rotated,rotated,Val(false),0.,1e-12,Int32(8192)).hit
    rotated_cross=(rotated[1],rotated[2],oblique(0,1e-9),oblique(.01,.01-1e-9))
    @test !ccd_planar_static_separated(rotated_cross,rotated_cross,Val(false),1e-16)
    @test tight_ccd(rotated_cross,rotated_cross,Val(false),0.,1e-12,Int32(8192)).hit
end

@testset "minimum separation uses a sphere rather than an axis-aligned cube" begin
    for T in (Float32, Float64)
        p = zero(SVector{3, T})
        for (factor, expected) in ((.8, false), (.6, true))
            q = SVector{3, T}(0, factor * 1e-4, factor * 1e-4)
            x = (p, p, q, q)
            result = tight_ccd(x, x, Val(false), T(1e-4), T(1e-10), Int32(8192))
            @test result.hit == expected && result.resolved
        end
        q = SVector{3, T}(0, .8e-4, .8e-4)
        result = tight_ccd((p, p, q, q), (p, p, p, p), Val(false), T(1e-4), T(1e-10), Int32(8192))
        @test result.hit && 0 < result.time < 1 - inv(sqrt(T(1.28)))
    end
end

@testset "spatial static certificates agree with exact rational geometry" begin
    for k in 1:512
        scale = exp2(mod(k * 17, 601) - 300)
        x = ntuple(i -> SVector{3, Float64}(sin(i*k), cos((i+1)*k), sin((i+2)*k)) * scale, 4)
        p, q, r, a = map(v -> Rational{BigInt}.(v), x)
        determinant = dot(p-a, cross(q-a, r-a))
        for kind in (Val(true), Val(false))
            normal = kind == Val(true) ? cross(r-q, a-q) : cross(q-p, a-r)
            for padding in (0., .001scale)
                certified = ccd_spatial_static_separated(x, x, kind, padding)
                @test !certified || determinant^2 > Rational{BigInt}(padding)^2 * dot(normal, normal)
            end
        end
    end
    v(x,y,z) = SVector{3,Float64}(x,y,z)
    point_face = (v(.0002,.0002,2e-8), v(0,0,0), v(.001,0,0), v(0,.001,0))
    skew_edges = (v(0,0,0), v(.001,0,0), v(.0005,-.001,2e-8), v(.0005,.001,2e-8))
    for (x, kind) in ((point_face, Val(true)), (skew_edges, Val(false)))
        @test ccd_spatial_static_separated(x, x, kind, 1e-16)
        @test !tight_ccd(x, x, kind, 0., 1e-12, Int32(8192)).hit
        @test !ccd_spatial_static_separated(x, x, kind, 1e-4)
        y = ntuple(i -> x[i] + v(0,0,-1e-8), 4)
        @test !ccd_spatial_static_separated(x, y, kind, 0.)
    end
    touching = (v(.0002,.0002,0), point_face[2], point_face[3], point_face[4])
    @test !ccd_spatial_static_separated(touching, touching, Val(true), 0.)
    @test tight_ccd(touching, touching, Val(true), 0., 1e-12, Int32(8192)).hit
end
