using PrintSim,Test
import PrintSim as PS

@testset "atomic extrema syntax lowers to modify, never load/store" begin
    for operation in (:min,:max)
        expression=:(PS.Atomix.@atomic $operation(a[1],b))
        expanded=string(macroexpand(@__MODULE__,expression))
        @test occursin("modify!",expanded)
        @test !occursin("set!",expanded)
    end
    directory=joinpath(dirname(pathof(PS)),"dynamics")
    for path in readdir(directory;join=true)
        endswith(path,".jl") || continue
        @test !occursin(r"Atomix\.@atomic[^\n]*=\s*(min|max)\(",read(path,String))
    end
end
