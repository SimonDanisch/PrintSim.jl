"Linux pidfd watchdog: pins the exact worker, never a potentially recycled PID."
function gpu_window_watchdog(pid::Integer,seconds::Real)
    pid>1 && isfinite(seconds) && 0<seconds<=900 || throw(ArgumentError("invalid GPU watchdog target/deadline"))
    script="import os,signal,sys,time; fd=os.pidfd_open(int(sys.argv[1])); time.sleep(float(sys.argv[2])); signal.pidfd_send_signal(fd,signal.SIGKILL)"
    return run(`python3 -c $script $pid $seconds`;wait=false)
end

"""
    with_gpu_window(f; seconds=900, lock_path="/tmp/bosgame-gpu.lock", device_selector=nothing)

Run `f(device, backend)` in a bounded, exclusively locked GPU window.
`device_selector` selects a Mantle device explicitly on multi-GPU workers;
`nothing` preserves the default device. The lock path is local to that worker.
On shared hardware the caller coordinates with other GPU users first; a
dedicated worker does not require external handoffs.
The lock covers device initialization, compilation, execution, cleanup and
final synchronization. A pidfd SIGKILL watchdog backs up the work deadline.
`existing_device` reuses a caller-owned live context for resident continuation;
do not also pass `device_selector`, or reuse it after any device loss/reset.
SIGKILL cannot retire a process stuck in an uninterruptible kernel wait; that
requires administrator device recovery, and must never be called a release.
"""
function with_gpu_window(f;seconds=900,lock_path="/tmp/bosgame-gpu.lock",host_memory_bytes=nothing,
        device_selector=nothing,existing_device=nothing)
    Sys.islinux() || error("the shared Radeon runner requires Linux flock")
    0<seconds<=900 || throw(ArgumentError("GPU windows must be bounded to at most 15 minutes"))
    device_selector===nothing || device_selector isa Mantle.DeviceSelector ||
        throw(ArgumentError("invalid Mantle GPU device selector"))
    existing_device===nothing || existing_device isa Mantle.Device ||
        throw(ArgumentError("existing_device must be a live Mantle device"))
    existing_device===nothing || device_selector===nothing ||
        throw(ArgumentError("select a device OR reuse a live device, not both"))
    lock=open(lock_path,"a+")
    device=nothing
    watchdog=nothing
    acquired=false
    failed=false
    synchronized=false
    old_data_limit=nothing
    try
        if host_memory_bytes!==nothing
            host_memory_bytes isa Integer && 0<host_memory_bytes<=typemax(UInt64) ||
                throw(ArgumentError("positive integer host-memory bound required"))
            limits=Ref((UInt64(0),UInt64(0)))
            # Linux RLIMIT_DATA covers the Julia private heap, not device-backed
            # shared mappings. Fail an excessive graph allocation locally rather
            # than letting the global OOM killer restart unrelated workers.
            ccall(:getrlimit,Cint,(Cint,Ref{NTuple{2,UInt64}}),2,limits)==0 ||
                error("cannot read worker host-memory limit")
            old_data_limit=limits[]
            bound=min(UInt64(host_memory_bytes),limits[][1],limits[][2])
            ccall(:setrlimit,Cint,(Cint,Ref{NTuple{2,UInt64}}),2,Ref((bound,limits[][2])))==0 ||
                error("cannot bound worker host-memory allocations")
        end
        ccall(:flock,Cint,(Cint,Cint),Base.fd(lock),6)==0 ||
            error("GPU reservation lock busy; device setup was not started")
        acquired=true
        watchdog=gpu_window_watchdog(getpid(),seconds)
        device=existing_device===nothing ?
            (device_selector===nothing ? Mantle.Device() : Mantle.Device(device_selector)) : existing_device
        return f(device,Mantle.backend(device))
    catch err
        failed=true
        device===nothing || @error "GPU window failed before final synchronization" exception=(err,catch_backtrace())
        rethrow()
    finally
        try
            device===nothing || GC.gc(false)
            try
                device===nothing || Mantle.waitidle(device)
                synchronized=device!==nothing
                # Workers are retired with SIGKILL after handoff, so their
                # atexit hook cannot persist the existing driver pipeline cache.
                # Save it while the synchronized device and shared lock live.
                if synchronized
                    try
                        Mantle.save_pipeline_cache!(device.ctx)
                    catch err
                        @warn "GPU synchronized, but optional pipeline-cache save failed" exception=(err,catch_backtrace())
                    end
                end
            catch err
                # A lost device can also fail cleanup. Keep the original
                # failing phase/stack rather than replacing it with waitidle.
                failed || rethrow()
                @error "GPU cleanup also failed" exception=(err,catch_backtrace())
            end
        finally
            watchdog===nothing || !process_running(watchdog) || kill(watchdog)
            acquired && ccall(:flock,Cint,(Cint,Cint),Base.fd(lock),8)
            close(lock)
            if old_data_limit!==nothing
                ccall(:setrlimit,Cint,(Cint,Ref{NTuple{2,UInt64}}),2,Ref(old_data_limit))==0 ||
                    @error "could not restore worker host-memory limit"
            end
            if acquired
                println(!failed && synchronized ? "RADEON_WINDOW_COMPLETE_LOCK_RELEASED" : "RADEON_WINDOW_FAILED_LOCK_RELEASED")
                flush(stdout)
            end
        end
    end
end
