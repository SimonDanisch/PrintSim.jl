using PrintSim, Test
import PrintSim as PS
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

@testset "host incidence preserves element and corner order" begin
    elements = [(3, 1, 3), (2, 4), (1,)]
    starts, ids, corners = PS.incidence(5, elements)
    for vertex in 1:5
        expected = [(Int32(e), Int32(k - 1)) for (e, el) in enumerate(elements) for (k, i) in enumerate(el) if i == vertex]
        range = starts[vertex]:(starts[vertex + 1] - 1)
        @test collect(zip(ids[range], corners[range])) == expected
    end
    @test PS.incidence(0, NTuple{3, Int32}[]) == (Int32[1], Int32[], Int32[])
    large = [(Int32(i), Int32(i + 1)) for i in 1:100000]
    starts, ids, corners = PS.incidence(100001, large)
    @test starts[end] == 200001
    @test ids[starts[50000]:(starts[50001] - 1)] == [49999, 50000]
    @test corners[starts[50000]:(starts[50001] - 1)] == [1, 0]
end

@testset "contact block scan handles empty, full and partial groups" begin
    for n in (0,1,255,256,257,1025), pattern in (:empty,:full,:alternating)
        a=Int32[pattern==:full || (pattern==:alternating && isodd(i)) ? 1 : 0 for i in 1:n]
        inc=PS.BodyIncidence(BACKEND,n,2;capacity=max(n,1))
        copyto!(inc.a,a)
        # Duplicate endpoints must only enter the incidence list once.
        copyto!(inc.b,a)
        expected=findall(==(1),a)
        for repeat in 1:2
            PS.compact!(BACKEND,inc)
            counts=Array(inc.total)
            @test counts==[length(expected),0]
            @test Array(inc.ids)[1:counts[1],1]==expected
            @test all(iszero,Array(inc.overflow))
        end
    end
end

@testset "deterministic contact incidence on the device" begin
    a = Int32[mod(i, 7) == 0 ? 1 : mod(i, 9) == 0 ? 2 : 0 for i in 1:777]
    b = Int32[mod(i, 11) == 0 ? 2 : mod(i, 13) == 0 ? 1 : 0 for i in 1:777]
    inc = PS.BodyIncidence(BACKEND, length(a), 3)
    copyto!(inc.a, a); copyto!(inc.b, b)
    PS.compact!(BACKEND, inc)
    ids, counts = Array(inc.ids), Array(inc.total)
    for body in 1:3
        expected = findall(i -> a[i] == body || b[i] == body, eachindex(a))
        @test ids[1:counts[body], body] == expected
    end
    @test all(iszero, Array(inc.overflow))
    small = PS.BodyIncidence(BACKEND, length(a), 3; capacity = 3)
    copyto!(small.a, a); copyto!(small.b, b)
    PS.compact!(BACKEND, small)
    @test Array(small.overflow)[1] > 0
    @test Array(small.total)[1] == 3
end
