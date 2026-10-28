using PrintSim,Test
import PrintSim as PS

@testset "GPU reservation guards reject before device setup" begin
    entered=Ref(false)
    callback=(device,backend)->(entered[]=true)
    @test_throws ArgumentError PS.with_gpu_window(callback;seconds=0)
    @test_throws ArgumentError PS.with_gpu_window(callback;seconds=901)
    @test_throws ArgumentError PS.with_gpu_window(callback;device_selector=1.5)
    @test_throws ArgumentError PS.with_gpu_window(callback;existing_device=1)
    mktemp() do path,owner
        @test ccall(:flock,Cint,(Cint,Cint),Base.fd(owner),6)==0
        @test_throws ErrorException PS.with_gpu_window(callback;seconds=1,lock_path=path)
        open(path,"a+") do observer
            @test ccall(:flock,Cint,(Cint,Cint),Base.fd(observer),6)!=0
        end
        ccall(:flock,Cint,(Cint,Cint),Base.fd(owner),8)
    end
    @test !entered[]
end

@testset "GPU watchdog pins and bounds only its target process" begin
    @test_throws ArgumentError PS.gpu_window_watchdog(1,1)
    @test_throws ArgumentError PS.gpu_window_watchdog(getpid(),0)
    child=run(`sleep 60`;wait=false)
    try
        watcher=PS.gpu_window_watchdog(getpid(child),.1)
        wait(watcher)
        wait(child)
        @test success(watcher)
        @test !process_running(child)
        @test !success(child)
    finally
        process_running(child) && kill(child)
    end
    child=run(`sleep 60`;wait=false)
    try
        watcher=PS.gpu_window_watchdog(getpid(child),60)
        kill(watcher)
        wait(watcher)
        @test process_running(child)
    finally
        process_running(child) && kill(child)
        wait(child)
    end
end
