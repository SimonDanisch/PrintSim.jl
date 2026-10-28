# The printed surface on the device against known answers: a square loop of beads, five layers
# high, is a square tube when outer walls bound its inside and a solid block when inner walls
# or infill do; its volume and area follow from the beads' capsules exactly (round outer
# corners, sharp inner ones). The rigid motion between two point sets and the move into the
# assembly against a rotation and shrink given in advance.
using PrintSim, Test, LinearAlgebra
import PrintSim as PS
import Mantle, GPUMeshing
using GeometryBasics: Vec3d, Mat3d

"""
Beads of width `w` around the square of side `a` centred on `(20, 20)`, at layers 0.2 high up
to 1.0, the loop of kind `kind`; `extra` adds pieces `(ax, ay, bx, by, kind)` to every layer.
"""
function squarebeads(a, kind; w = 0.4f0, extra = ())
    lo, hi = 20f0 - a / 2, 20f0 + a / 2
    loop = [(lo, lo, hi, lo), (hi, lo, hi, hi), (hi, hi, lo, hi), (lo, hi, lo, lo)]
    pieces = [(p..., kind) for p in loop]
    append!(pieces, extra)
    nl = 5
    ends = reduce(hcat, [Float32[p[1], p[2], p[3], p[4]] for l in 1:nl for p in pieces])
    kinds = Int32[p[5] for l in 1:nl for p in pieces]
    layer = Int32[l for l in 1:nl for _ in pieces]
    layers = Float32[0.2f0 * (l - 1) for _ in 1:1, l in 1:nl]
    return PS.BeadSegments(ends, fill(Float32(w), length(kinds)), layer, kinds, vcat(layers, layers .+ 0.2f0))
end

function printedsurfacetests(dev)
    a, r, h = 10.0, 0.2, 1.0
    outer = (a + 2r)^2 - (4 - π) * r^2
    band = outer - (a - 2r)^2
    rim = 4a + 2π * r

    @testset "bead kinds" begin
        @test PS.bead_kind("Outer wall") == PS.BEAD_OUTER
        @test PS.bead_kind("Overhang wall") == PS.BEAD_OUTER
        @test PS.bead_kind("Inner wall") == PS.BEAD_INNER
        @test PS.bead_kind("Sparse infill") == PS.BEAD_FILL
        @test PS.bead_kind("Bridge") == PS.BEAD_FILL
    end

    @testset "outer walls around a void: a square tube" begin
        V, F = printed_surface(dev, squarebeads(a, PS.BEAD_OUTER))
        @test GPUMeshing.closedmanifold(dev, F)
        area, vol = GPUMeshing.areavolume(dev, V, F)
        @test vol ≈ band * h rtol = 1e-3
        @test area ≈ (rim + 4(a - 2r)) * h + 2band rtol = 2e-3
        foreach(Mantle.free!, (V, F))
    end

    @testset "inner walls or infill around a void: a solid block" begin
        for beads in (squarebeads(a, PS.BEAD_INNER),
                      squarebeads(a, PS.BEAD_OUTER; extra = ((15.2f0, 20f0, 24.8f0, 20f0, PS.BEAD_FILL),)))
            V, F = printed_surface(dev, beads)
            @test GPUMeshing.closedmanifold(dev, F)
            area, vol = GPUMeshing.areavolume(dev, V, F)
            @test vol ≈ outer * h rtol = 1e-3
            @test area ≈ rim * h + 2outer rtol = 2e-3
            foreach(Mantle.free!, (V, F))
        end
    end

    @testset "parts simplified together, each within the tolerance of its own" begin
        beads = [squarebeads(a, PS.BEAD_OUTER), squarebeads(a, PS.BEAD_INNER)]
        full = [size(printed_surface(dev, b)[2], 2) for b in beads]
        meshes = printed_surfaces(dev, beads; tolerance = 0.014)
        for ((V, F), n, expected) in zip(meshes, full, (band, outer))
            @test GPUMeshing.closedmanifold(dev, F)
            @test size(F, 2) < n ÷ 10
            _, vol = GPUMeshing.areavolume(dev, V, F)
            @test vol ≈ expected * h rtol = 2e-2
        end
    end

    @testset "rigid motion by Kabsch's method, and the move into the assembly" begin
        M = Mat3d(Matrix(qr([1.0 0.3 -0.2; -0.4 1 0.1; 0.2 0.5 1]).Q))
        det(M) < 0 && (M = Mat3d(M * Diagonal([1, 1, -1])))
        t = Vec3d(120.0, -35.0, 12.5)
        P = Float32[sin(3i + k) * (20 + 5k) for k in 1:3, i in 1:1000]
        S = Float32.(Matrix(M) * Float64.(P) .+ t)
        R, u, err, height = PS.rigidmotion(Mantle.Buffer(dev, P), Mantle.Buffer(dev, S))
        @test Matrix(R) ≈ Matrix(M) atol = 1e-5
        @test u ≈ t atol = 1e-3
        @test err < 1e-3
        @test height ≈ maximum(S[3, :]) - minimum(S[3, :]) rtol = 1e-6
        q = PS.PartPlacement(101, 1, M, t, Vec3d(100.0, 50.0, 0.0), 0.003)
        X = Float32[110 95 130; 40 60 52; 2 7 0.5]
        Y = Array(Mantle.storage(assembled!(dev, Mantle.Buffer(dev, X), q)))
        for i in 1:3
            p = Float64.(X[:, i])
            cooled = Vec3d(100 + (p[1] - 100) * 0.997, 50 + (p[2] - 50) * 0.997, p[3])
            @test Y[:, i] ≈ Matrix(M)' * (cooled - t) * 1e-3 atol = 1e-8
        end
    end
end

printedsurfacetests(TEST_DEVICE)
