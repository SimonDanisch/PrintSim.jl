using Test
include("facets.jl")

@testset "a sheet on a faceted roller" begin
    # Both coarse facets and a fine mesh must retain face contacts without chattering or
    # passing through the sheet. The full two-grid study is run separately via facets.jl.
    for (N, n) in ((32, 17), (256, 33))
        result = facet_run(BACKEND, N, 2e10; n)
        # The same 0.5 % load bound as the stack tests, with positive sampled surface clearance.
        @test maximum(abs(l - 1) for l in result.late_load) < 0.005
        @test result.clearance_um > 0
    end
end
