using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

function atomic_extrema_fixture!(minimum,maximum,values)
    i=PS.KI.get_global_id(Int32).x
    i<=length(values) || return nothing
    @inbounds begin
        PS.Atomix.@atomic min(minimum[1],values[i])
        PS.Atomix.@atomic max(maximum[1],values[i])
    end
    return nothing
end

@testset "contended extrema use atomic read-modify-write across workgroups" begin
    n=8193
    values=UInt32[mod(7919i,n) for i in 1:n]
    minimum=PS.upload(BACKEND,UInt32[typemax(UInt32)])
    maximum=PS.upload(BACKEND,UInt32[0])
    input=PS.upload(BACKEND,values)
    for _ in 1:8
        copyto!(minimum,PS.upload(BACKEND,UInt32[typemax(UInt32)]))
        PS.launch!(BACKEND,PS.zero!,(maximum,),1)
        PS.launch!(BACKEND,atomic_extrema_fixture!,(minimum,maximum,input),n)
        @test only(Array(minimum))==0
        @test only(Array(maximum))==n-1
    end
    # The physical Newton monitor must not lose a large correction in a
    # different workgroup, even when most nodes do not move at all.
    base=fill(zero(PS.Position),n)
    x=copy(base)
    x[4097]=PS.Position(PS.Vec3f(1f-3,0,0))
    dx=PS.upload(BACKEND,x)
    db=PS.upload(BACKEND,base)
    for _ in 1:8
        PS.launch!(BACKEND,PS.zero!,(maximum,),1)
        PS.launch!(BACKEND,PS.newton_step_max!,(maximum,dx,db),n)
        @test reinterpret(Float32,only(Array(maximum)))==1f-3
    end
    x[4097]=PS.Position(PS.Vec3f(NaN32,0,0))
    copyto!(dx,PS.upload(BACKEND,x))
    PS.launch!(BACKEND,PS.zero!,(maximum,),1)
    PS.launch!(BACKEND,PS.newton_step_max!,(maximum,dx,db),n)
    @test reinterpret(Float32,only(Array(maximum)))==Inf32
end
