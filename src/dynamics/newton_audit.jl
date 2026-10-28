# CFL-style trial-motion cap (StiffGIPC GIPC.cu, alpha_CFL). This is a line
# search heuristic, NOT a collision certificate or a convergence criterion.
# Keep the unclipped correction untouched. Eight is conservative headroom for
# Euclidean/proxy interpolation amplification of the nodal component maximum.
@inline function newton_motion_fraction(maximum_step::Float32,limit::Float32)
    limit>0 || return 1f0
    isfinite(maximum_step) && maximum_step>=0 || return 0f0
    maximum_step==0 && return 1f0
    return min(1f0,limit/(8f0*maximum_step))
end

function newton_motion_bound!(bound,maximum_step,limit::Float32)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds begin
        step=reinterpret(Float32,maximum_step[1])
        cap=newton_motion_fraction(step,limit)
        Atomix.@atomic min(bound[1],reinterpret(UInt32,cap))
    end
    return nothing
end

function newton_step_max!(maximum_step,x,base)
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds begin
        delta=x[i]-base[i]
        value = maximum(abs,delta)
        all(isfinite,delta) && isfinite(value) || (value = Inf32)
        Atomix.@atomic max(maximum_step[1],reinterpret(UInt32,value))
    end
    return nothing
end

function affine_newton_step_max!(maximum_step,q,base,model,C,one)
    i = KI.get_global_id(Int32).x
    i <= length(C.point_owner) || return nothing
    @inbounds begin
        body = model.mesh_body[C.point_owner[i]]
        body == 0 && return nothing
        u = model.coordinates[i]
        step = affine_point(q,body,u,one[1])-affine_point(base,body,u,one[1])
        value = maximum(abs,step)
        all(isfinite,step) && isfinite(value) || (value = Inf32)
        Atomix.@atomic max(maximum_step[1],reinterpret(UInt32,value))
    end
    return nothing
end

function initialize_newton_progress!(E,history)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds begin
        E.newton_status[1]=1
        E.newton_status[2]=0
        for j in axes(E.newton_progress,2),i in 1:5
            E.newton_progress[i,j]=0
        end
        if history!==nothing
            for j in axes(history,2)
                history[1,j]=0
                history[2,j]=0
                history[3,j]=0
                history[4,j]=1 # inactive Newton slots are exact zero solves
            end
        end
    end
    return nothing
end

function record_newton_progress!(E,iteration::Int32,tolerance::Float32)
    KI.get_global_id(Int32).x == 1 || return nothing
    @inbounds begin
        E.newton_progress[1,iteration] = reinterpret(Float32,E.newton_step[1])
        E.newton_progress[2,iteration] = reinterpret(Float32,E.bound[1])
        E.newton_progress[3,iteration] = Float32(E.accepted[1])
        E.newton_progress[4,iteration] = E.sum[1]
        E.newton_progress[5,iteration] = E.sum[2]
        E.newton_status[2]=UInt32(iteration)
        if tolerance>0 && reinterpret(Float32,E.newton_step[1])<=tolerance && E.accepted[1]!=0 && E.cg_active[1]==0
            E.newton_status[1]=0
        end
    end
    return nothing
end

function record_dynamic_newton_progress!(E,tolerance::Float32)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds iteration=Int32(E.newton_status[2])+Int32(1)
    record_newton_progress!(E,iteration,tolerance)
    return nothing
end
