using PrintSim, Test, LinearAlgebra
import PrintSim as PS
import GeometryBasics as GB

@testset "only perimeter-bounded holes survive infill filling" begin
    G = PS.GEOS
    path(points; width = 0.4) = PS.BeadPath(PS.Vec2d.(points), 0.2, 0.2, width, true)
    outer = path([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.06)])
    inner = path([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0), (1.0, 1.06)])
    region = PS.layer_region([outer, inner])
    @test length(PS.outer_wall_loops([outer, inner])) == 2
    @test G.covers(region, G.Point(0.5, 0.5))
    @test !G.covers(region, G.Point(2.0, 2.0))
    @test !G.covers(region, G.Point(-0.3, 0.0))
    # Width changes split a perimeter into BeadPaths; they do not open its bore.
    a = path([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)])
    c = path([(4.0, 4.0), (0.0, 4.0), (0.0, 0.06)]; width = 0.42)
    split = PS.layer_region([a, c, inner])
    @test G.covers(split, G.Point(0.5, 0.5))
    @test !G.covers(split, G.Point(2.0, 2.0))
    island = path([(1.4, 1.4), (2.6, 1.4), (2.6, 2.6), (1.4, 2.6), (1.4, 1.4)])
    with_island = PS.layer_region([outer, inner, island])
    @test G.covers(with_island, G.Point(2.0, 2.0))
end

@testset "rounded layer heights do not create nanometer slivers" begin
    G = PS.GEOS
    a = G.readgeom("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))")
    b = G.readgeom("POLYGON ((0 0, 4 0, 4 4, 2 4.1, 0 4, 0 0))")
    item = Item("rounded height seam", :PLA, :gray, _ -> nothing)
    layers = [PS.PrintedLayer(z0, z1, r, G.prepareGeom(r)) for (z0, z1, r) in
        ((0.0, 0.199999, a), (0.199999, 0.2, b), (0.2, 0.4, a))]
    part = PrintedPart(item, layers, one(PS.Mat3d), PS.Vec3d(0), PS.Vec3d(0), 0.0, 0.0)
    mesh = printed_mesh(part)
    counts = Dict{Tuple{Int, Int}, Int}()
    for face in GB.faces(mesh), k in 1:3
        edge = minmax(convert(Int, face[k]), convert(Int, face[mod1(k + 1, 3)]))
        counts[edge] = get(counts, edge, 0) + 1
    end
    @test all(==(2), values(counts))
    @test all(p -> any(z -> isapprox(p[3], z; atol = 1e-15),
        (0.0, 0.0001, 0.0002, 0.0003, 0.0004)), GB.coordinates(mesh))
    builder = ModelBuilder()
    add_body!(TEST_DEVICE, builder, mesh)
    @test PS.volume_properties(TEST_DEVICE, builder.meshes[1])[1] ≈ 6.4e-9
end

function synthetic_print(regions; shrink = 0.01)
    item = Item("synthetic layered solid", :PLA, :gray, _ -> nothing)
    layers = [PS.PrintedLayer(0.2 * (i - 1), 0.2 * i, region, PS.GEOS.prepareGeom(region)) for (i, region) in enumerate(regions)]
    return PrintedPart(item, layers, one(PS.Mat3d), PS.Vec3d(0), PS.Vec3d(0), shrink, 0.0)
end

@testset "combined infill does not double the exterior layer height" begin
    item = Item("combined infill", :PLA, :gray, _ -> nothing)
    axes = (PS.Vec3d(1, 0, 0), PS.Vec3d(0, 1, 0), PS.Vec3d(0, 0, 1))
    mesh = box_mesh(PS.Vec3d(0), axes, (1.0, 1.0, 0.2))
    object = PS.PrintObject(item, mesh, PS.Vec3d(2, 2, 0.4), one(PS.Mat3d), PS.Vec3d(0))
    placement = PS.Placement(1, PS.Point2d(10, 10), false)
    loop = PS.Vec2d[PS.Vec2d(9, 9), PS.Vec2d(11, 9), PS.Vec2d(11, 11), PS.Vec2d(9, 11), PS.Vec2d(9, 9)]
    paths = [PS.BeadPath(loop, z, 0.2, 0.4, true) for z in (0.2, 0.4, 0.6)]
    push!(paths, PS.BeadPath(PS.Vec2d[PS.Vec2d(9, 10), PS.Vec2d(11, 10)], 0.4, 0.4, 0.4, false))
    part = PrintedPart(object, placement, paths; shrink = 0.0)
    @test length(part.layers) == 3
    @test part.layers[2].z0 ≈ 0.2
    @test all(l -> l.z1 - l.z0 ≈ 0.2, part.layers)
    @test all(i -> part.layers[i].z0 == part.layers[i - 1].z1, 2:length(part.layers))
end

function surface_edges(mesh)
    edges = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    for f in GB.faces(mesh), k in 1:3
        a, b = convert(Int, f[k]), convert(Int, f[mod1(k + 1, 3)])
        count, orientation = get(edges, minmax(a, b), (0, 0))
        edges[minmax(a, b)] = count + 1, orientation + (a < b ? 1 : -1)
    end
    return edges
end

@testset "printed layer boundary has no internal faces or T-junctions" begin
    G = PS.GEOS
    a = G.readgeom("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 3, 3 3, 3 1, 1 1))")
    b = G.readgeom("POLYGON ((0.5 0, 4.5 0, 4.5 4, 0.5 4, 0.5 0), (1.5 1, 1.5 3, 3.5 3, 3.5 1, 1.5 1))")
    c = G.readgeom("MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 4 0, 4 1, 3 1, 3 0)))")
    d = G.readgeom("POLYGON ((0 0, 4 0, 4 1, 3 1, 3 2, 0 2, 0 0))")
    for regions in ([a, a], [a, b, a], [c, d, c])
        part = synthetic_print(regions)
        mesh = printed_mesh(part)
        @test all(==((2, 0)), values(surface_edges(mesh)))
        builder = ModelBuilder()
        id = add_body!(TEST_DEVICE, builder, mesh)
        volume, _, _ = PS.volume_properties(TEST_DEVICE, builder.meshes[id])
        exact = sum(G.area(region) * 0.2 for region in regions) * 0.99^2 * 1e-9
        @test volume ≈ exact rtol = 1e-8
        @test all(f -> begin
            x, y, z = (GB.coordinates(mesh)[i] for i in f)
            norm(cross(y - x, z - x)) > 0
        end, GB.faces(mesh))
    end
end
