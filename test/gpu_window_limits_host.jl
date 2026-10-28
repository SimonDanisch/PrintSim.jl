using PrintSim,Test
import PrintSim as PS

@testset "bounded worker heap is restored when GPU lock is unavailable" begin
    if Sys.islinux()
        limits()=begin
            value=Ref((UInt64(0),UInt64(0)))
            @test ccall(:getrlimit,Cint,(Cint,Ref{NTuple{2,UInt64}}),2,value)==0
            value[]
        end
        original=limits()
        mktemp() do path,holder
            @test ccall(:flock,Cint,(Cint,Cint),Base.fd(holder),6)==0
            called=Ref(false)
            err=try
                PS.with_gpu_window(;seconds=1,lock_path=path,host_memory_bytes=128*1024^3) do _,_
                    called[]=true
                end
                nothing
            catch err
                err
            end
            @test err isa ErrorException && occursin("lock busy",sprint(showerror,err))
            @test !called[]
            @test limits()==original
            @test ccall(:flock,Cint,(Cint,Cint),Base.fd(holder),8)==0
        end
        @test_throws ArgumentError PS.with_gpu_window((_,_)->nothing;host_memory_bytes=-1)
        @test limits()==original
    end
end
