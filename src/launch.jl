# Kernels are plain functions over KernelInterface's intrinsics; `launch!` runs one now or
# declares it into a Mantle graph.

"""
    launch!(target, kernel, args, n; group = 256)

Runs `kernel(args...)` over `n` items, in groups of `group`: now on a backend, or as a node of
a Mantle graph, which runs it with the graph. A kernel takes its item from
`KI.get_global_id(Int32).x` and returns for items past `n`: the last group is padded.
"""
function launch!(b::KI.Backend, kernel, args::Tuple, n; group = 256)
    n <= 0 && return nothing
    tt = Tuple{map(a -> Core.Typeof(KI.argconvert(b, a)), args)...}
    KI.kernel_function(b, kernel, tt)(args...; ndrange = n, workgroupsize = group)
    return nothing
end
launch!(g::Mantle.Graph, kernel, args::Tuple, n; group = 256) = (n <= 0 || Mantle.dispatch!(g, kernel, args, n; group, name = string(nameof(kernel))); nothing)

"Launch bounded query batches; the kernel takes trailing (first, count) and guards padded lanes."
function launch_indexed_batches!(target,kernel,args::Tuple,n;batch=8192,group=256)
    batch>0 && 0<=n<=typemax(Int32) || throw(ArgumentError("invalid indexed dispatch range"))
    for first in 1:batch:n
        count=min(batch,n-first+1)
        launch!(target,kernel,(args...,Int32(first),Int32(count)),count;group)
    end
    return nothing
end

"Gate a recorded phase with a device scalar without nesting Mantle CG loops."
struct ConditionalTarget{T,A}
    target::T
    active::A
end

@inline function conditional_kernel!(active,::Val{K},args...) where {K}
    @inbounds active[1]==0 && return nothing
    K(args...)
    return nothing
end

function launch!(target::ConditionalTarget,kernel,args::Tuple,n;group=256)
    # Keep buffer arguments flat. A nested tuple containing heterogeneous
    # model-array pointers currently defeats SPIR-V aggregate legalization.
    launch!(target.target,conditional_kernel!,(target.active,Val(kernel),args...),n;group)
end

is_recording(target)=target isa Mantle.Graph
is_recording(target::ConditionalTarget)=is_recording(target.target)
recording_target(target)=target
recording_target(target::ConditionalTarget)=recording_target(target.target)
