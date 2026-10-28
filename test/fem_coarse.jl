using PrintSim, Test, LinearAlgebra
import PrintSim as PS

@testset "coarse factor preserves full finite-element energy" begin
    b = ModelBuilder()
    points, tets = tet_box((0.008, 0.006, 0.002), (3, 3, 3))
    add_deformable!(b, points, tets)
    dt = 1e-3
    model = PS.FEMModel(Array, b)
    coarse = PS.FEMCoarse(Array, b, dt)
    modes = only(coarse.mode_count)
    n = 3modes
    first = only(coarse.factor)
    L = reshape(coarse.scratch[first:(first + n^2 - 1)], n, n)
    P = transpose(coarse.basis[1:model.nodes, 1:modes])
    for sample in 1:6
        coefficients = sin.(collect(1:n) .* (sample / 7))
        displacement = reshape(coefficients, 3, modes) * P
        energy = sum(model.mass[i] * sum(abs2, displacement[:, i]) / dt^2 for i in 1:model.nodes)
        for e in 1:model.elements
            gradient = zeros(3, 3)
            for corner in 1:10
                node = model.tets[corner, e]
                g = Float64.(model.gradient_hi[corner, e]) + Float64.(model.gradient_lo[corner, e])
                gradient .+= displacement[:, node] * g'
            end
            strain = (gradient[1, 1], gradient[2, 2], gradient[3, 3],
                (gradient[1, 2] + gradient[2, 1]) / sqrt(2),
                (gradient[1, 3] + gradient[3, 1]) / sqrt(2),
                (gradient[2, 3] + gradient[3, 2]) / sqrt(2), tr(gradient))
            energy += sum(model.stiffness[k, e] * strain[k]^2 for k in 1:7)
        end
        @test sum(abs2, L' * coefficients) ≈ energy rtol = 5e-6
    end
end

@testset "parallel coarse restriction matches the Galerkin projection" begin
    isdefined(@__MODULE__, :BACKEND) || include("backend.jl")
    b = ModelBuilder()
    for (size, dims) in (((0.008, 0.006, 0.002), (6, 5, 4)),
            ((0.003, 0.004, 0.001), (3, 3, 2)))
        points, tets = tet_box(size, dims)
        add_deformable!(b, points, tets)
    end
    coarse = PS.FEMCoarse(BACKEND, b, 1e-3)
    host = PS.Adapt.adapt(Array, coarse)
    nodes = length(host.node_part)
    r = PS.Vec3f[PS.Vec3f(sin(i / 7), cos(i / 11), sin(i / 13)) for i in 1:nodes]
    residual = PS.upload(BACKEND, r)
    tiles = Int32(cld(nodes, 256))
    for _ in 1:2 # clearing the prior projection is part of every application
        PS.launch!(BACKEND, PS.fem_coarse_clear!, (coarse,), coarse.parts)
        PS.launch!(BACKEND, PS.fem_coarse_restrict_tiled!, (coarse, residual, tiles), Int(tiles) * 256 * coarse.modes)
        actual = Array(coarse.scratch)
        for part in 1:coarse.parts, mode in 1:host.mode_count[part]
            first, count = host.node_first[part], host.node_count[part]
            expected = sum(Float64(host.basis[i, mode]) * Float64.(r[i]) for i in first:(first + count - 1))
            at = host.rhs[part] + 3(mode - 1)
            @test isapprox(actual[at:(at + 2)], expected; rtol = 3e-5, atol = 2e-6)
        end
    end
end
