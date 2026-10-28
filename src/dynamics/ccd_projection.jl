# A conservative separating-axis prefilter for the convex hull of the relative
# feature trajectories (Tight Inclusion, Wang et al. 2021, Proposition 5.1).
# This is not a replacement CCD solver: uncertainty falls through unchanged.
# VF uses the actual barycentric triangle's three corners, not the enclosing
# parameter rectangle's extra corner. EE uses all four endpoint differences.
# Arithmetic intervals enclose the exact stored hi+lo positions, not rounded
# approximate coordinates. No external implementation is copied here.

# Widen every elementary operation. Widen tiny results to a full normal unit
# as well, so flush-to-zero/subnormal arithmetic cannot create a certificate.
@inline ccd_projection_lower(v::Float64)=!isfinite(v) ? -Inf :
    abs(v)<floatmin(Float64) ? -floatmin(Float64) : prevfloat(v)
@inline ccd_projection_upper(v::Float64)=!isfinite(v) ? Inf :
    abs(v)<floatmin(Float64) ? floatmin(Float64) : nextfloat(v)

@inline function ccd_projection_wide(v::Float32)
    bits=reinterpret(UInt32,v)
    # Some devices flush Float32 subnormal inputs during conversion. Decline
    # certification for that component, rather than assuming preservation.
    bits & UInt32(0x7f800000)==0 && bits & UInt32(0x007fffff)!=0 && return NaN
    return Float64(v)
end

@inline function ccd_projection_difference(p::Position,q::Position,k)
    h=ccd_projection_wide(p.hi[k])-ccd_projection_wide(q.hi[k])
    l=ccd_projection_wide(p.lo[k])-ccd_projection_wide(q.lo[k])
    return (ccd_projection_lower(ccd_projection_lower(h)+ccd_projection_lower(l)),
        ccd_projection_upper(ccd_projection_upper(h)+ccd_projection_upper(l)))
end

@inline function ccd_projection_dot(p::Position,q::Position,axis)
    lower,upper=0.,0.
    for k in Int32(1):Int32(3)
        coefficient=axis[k]
        coefficient==0 && continue
        a,b=ccd_projection_difference(p,q,k)
        lo,hi=coefficient>0 ? (a,b) : (b,a)
        lower=ccd_projection_lower(lower+ccd_projection_lower(coefficient*lo))
        upper=ccd_projection_upper(upper+ccd_projection_upper(coefficient*hi))
    end
    return lower,upper
end

"Any finite axis is valid; computed normal/edge accuracy affects rejection rate only."
@inline function ccd_projection_pair_range(ccd,a,b,normal,prefix_scale)
    @inbounds lo0,hi0=ccd_projection_dot(ccd.start[a],ccd.start[b],normal)
    @inbounds lo1,hi1=ccd_projection_dot(ccd.finish[a],ccd.finish[b],normal)
    if prefix_scale!=1.
        # Internal prefixes are dyadic >=2^-32, so both coefficients below are
        # exact. Interval endpoints enclose the exact original trajectory at
        # prefix_scale; no rounded/interpolated Position is substituted.
        first=1.0 - prefix_scale
        lo1=ccd_projection_lower(ccd_projection_lower(first*lo0)+ccd_projection_lower(prefix_scale*lo1))
        hi1=ccd_projection_upper(ccd_projection_upper(first*hi0)+ccd_projection_upper(prefix_scale*hi1))
    end
    return min(lo0,lo1),max(hi0,hi1)
end

@inline function ccd_projection_axis_separated(ccd,ids,::Val{VF},separation,axis,prefix_scale=1.) where VF
    # Powers of two keep 1-prefix_scale exactly representable. Refuse other
    # caller inputs instead of assuming an unbounded coefficient is exact.
    isfinite(prefix_scale) && 2.0^-32<=prefix_scale<=1. &&
        reinterpret(UInt64,prefix_scale) & UInt64(0x000fffffffffffff)==0 || return false
    @inbounds for id in ids
        a,b=ccd.start[id],ccd.finish[id]
        all(isfinite,a.hi) && all(isfinite,a.lo) && all(isfinite,b.hi) && all(isfinite,b.lo) || return false
    end
    magnitude=maximum(abs,axis)
    isfinite(magnitude) && magnitude>0 && isfinite(separation) && separation>=0 || return false
    normal=axis/magnitude
    all(isfinite,normal) || return false
    norm_bound=ccd_projection_upper(ccd_projection_upper(abs(normal[1])+abs(normal[2]))+abs(normal[3]))
    wide_radius=separation isa Float32 ? ccd_projection_wide(separation) : Float64(separation)
    isfinite(wide_radius) || return false
    radius=separation==0 ? 0. : ccd_projection_upper(wide_radius*norm_bound)
    low,high=Inf,-Inf
    if VF
        for b in Int32(2):Int32(4)
            @inbounds lo,hi=ccd_projection_pair_range(ccd,ids[1],ids[b],normal,prefix_scale)
            low,high=min(low,lo),max(high,hi)
        end
    else
        for a in Int32(1):Int32(2),b in Int32(3):Int32(4)
            @inbounds lo,hi=ccd_projection_pair_range(ccd,ids[a],ids[b],normal,prefix_scale)
            low,high=min(low,lo),max(high,hi)
        end
    end
    # Every relative feature value is a convex combination of these six/eight
    # exact endpoint values. ||axis||_1 bounds ||axis||_2 for finite thickness.
    return low>radius || high < -radius
end

@inline ccd_projected_separated(ccd,ids,kind,separation,::Val{false})=false
@inline ccd_projected_separated(ccd,ids,kind,separation,::Val{true})=
    ccd_projected_certificate(ccd,ids,kind,separation,1.)
@inline function ccd_projected_certificate(ccd,ids,kind::Val{VF},separation,prefix_scale) where VF
    @inbounds origin=ccd.start[ids[1]]
    @inbounds x=ntuple(k->wide_ccd_relative(ccd.start[ids[k]],origin),Val(4))
    edge1=VF ? x[3]-x[2] : x[2]-x[1]
    edge2=VF ? x[4]-x[2] : x[4]-x[3]
    normal=cross(edge1,edge2)
    ccd_projection_axis_separated(ccd,ids,kind,separation,normal,prefix_scale) && return true
    # Coordinate projections of the start feature edges provide in-plane
    # separating axes for coplanar/near-parallel features. All are certificates
    # for the full trajectory, not assumptions of static or common motion.
    for edge in Int32(1):Int32(VF ? 3 : 2)
        direction=VF ? (edge==1 ? x[3]-x[2] : edge==2 ? x[4]-x[3] : x[2]-x[4]) :
            (edge==1 ? edge1 : edge2)
        for coordinate in Int32(1):Int32(3)
            axis=coordinate==1 ? Vec3d(0,direction[3],-direction[2]) :
                coordinate==2 ? Vec3d(-direction[3],0,direction[1]) :
                Vec3d(direction[2],-direction[1],0)
            ccd_projection_axis_separated(ccd,ids,kind,separation,axis,prefix_scale) && return true
        end
    end
    return false
end

@inline ccd_projected_prefix(ccd,ids,kind,separation,::Val{false})=0f0

# A linearly varying axis is also a valid separating axis at every time.
# Its dot product with a linear relative trajectory is quadratic. The three
# Bernstein coefficients are dot(r0,n0), (dot(r0,n1)+dot(r1,n0))/2, dot(r1,n1).
# Outward bounds on ALL coefficients therefore certify the entire trajectory,
# not sampled times. Axis endpoints are arbitrary finite computed vectors;
# their geometric accuracy affects rejection rate, never correctness.
@inline function ccd_projection_moving_pair_range(ccd,a,b,n0,n1)
    @inbounds begin
        l00,h00=ccd_projection_dot(ccd.start[a],ccd.start[b],n0)
        l01,h01=ccd_projection_dot(ccd.start[a],ccd.start[b],n1)
        l10,h10=ccd_projection_dot(ccd.finish[a],ccd.finish[b],n0)
        l11,h11=ccd_projection_dot(ccd.finish[a],ccd.finish[b],n1)
    end
    middle_lo=ccd_projection_lower(ccd_projection_lower(l01+l10)*.5)
    middle_hi=ccd_projection_upper(ccd_projection_upper(h01+h10)*.5)
    return min(l00,middle_lo,l11),max(h00,middle_hi,h11)
end

@inline function ccd_projection_moving_axis_separated(ccd,ids,::Val{VF},separation,axis0,axis1) where VF
    @inbounds for id in ids
        a,b=ccd.start[id],ccd.finish[id]
        all(isfinite,a.hi) && all(isfinite,a.lo) && all(isfinite,b.hi) && all(isfinite,b.lo) || return false
    end
    magnitude=max(maximum(abs,axis0),maximum(abs,axis1))
    isfinite(magnitude) && magnitude>0 && isfinite(separation) && separation>=0 || return false
    n0,n1=axis0/magnitude,axis1/magnitude
    all(isfinite,n0) && all(isfinite,n1) || return false
    # The L1 norm of a linear axis is bounded by its endpoint norms. This
    # upper-bounds the Euclidean norm required for finite-thickness clearance.
    norm0=ccd_projection_upper(ccd_projection_upper(abs(n0[1])+abs(n0[2]))+abs(n0[3]))
    norm1=ccd_projection_upper(ccd_projection_upper(abs(n1[1])+abs(n1[2]))+abs(n1[3]))
    wide_radius=separation isa Float32 ? ccd_projection_wide(separation) : Float64(separation)
    isfinite(wide_radius) || return false
    radius=separation==0 ? 0. : ccd_projection_upper(wide_radius*max(norm0,norm1))
    low,high=Inf,-Inf
    if VF
        for b in Int32(2):Int32(4)
            @inbounds lo,hi=ccd_projection_moving_pair_range(ccd,ids[1],ids[b],n0,n1)
            low,high=min(low,lo),max(high,hi)
        end
    else
        for a in Int32(1):Int32(2),b in Int32(3):Int32(4)
            @inbounds lo,hi=ccd_projection_moving_pair_range(ccd,ids[a],ids[b],n0,n1)
            low,high=min(low,lo),max(high,hi)
        end
    end
    return low>radius || high < -radius
end

@inline ccd_projected_moving_separated(ccd,ids,kind,separation,::Val{false})=false
@inline function ccd_projected_moving_separated(ccd,ids,kind::Val{VF},separation,::Val{true}) where VF
    @inbounds x=ntuple(k->wide_ccd_relative(ccd.start[ids[k]],ccd.start[ids[1]]),Val(4))
    @inbounds y=ntuple(k->wide_ccd_relative(ccd.finish[ids[k]],ccd.finish[ids[1]]),Val(4))
    e0=VF ? x[3]-x[2] : x[2]-x[1]
    f0=VF ? x[4]-x[2] : x[4]-x[3]
    e1=VF ? y[3]-y[2] : y[2]-y[1]
    f1=VF ? y[4]-y[2] : y[4]-y[3]
    ccd_projection_moving_axis_separated(ccd,ids,kind,separation,cross(e0,f0),cross(e1,f1)) && return true
    for edge in Int32(1):Int32(VF ? 3 : 2)
        d0=VF ? (edge==1 ? x[3]-x[2] : edge==2 ? x[4]-x[3] : x[2]-x[4]) : (edge==1 ? e0 : f0)
        d1=VF ? (edge==1 ? y[3]-y[2] : edge==2 ? y[4]-y[3] : y[2]-y[4]) : (edge==1 ? e1 : f1)
        for coordinate in Int32(1):Int32(3)
            n0=coordinate==1 ? Vec3d(0,d0[3],-d0[2]) : coordinate==2 ? Vec3d(-d0[3],0,d0[1]) : Vec3d(d0[2],-d0[1],0)
            n1=coordinate==1 ? Vec3d(0,d1[3],-d1[2]) : coordinate==2 ? Vec3d(-d1[3],0,d1[1]) : Vec3d(d1[2],-d1[1],0)
            ccd_projection_moving_axis_separated(ccd,ids,kind,separation,n0,n1) && return true
        end
    end
    return false
end

@inline function ccd_projected_prefix(ccd,ids,kind,separation,::Val{true})
    scale=.5
    for _ in Int32(1):Int32(16)
        ccd_projected_certificate(ccd,ids,kind,separation,scale) && return Float32(scale)
        scale*=.5
    end
    return 0f0
end

@inline record_ccd_primitive_bound!(::Nothing,bits)=nothing
@inline function record_ccd_primitive_bound!(witness::Tuple,bits)
    times,index=witness
    @inbounds Atomix.@atomic min(times[index],bits)
    return nothing
end

@inline function filtered_swept_accumulate!(ccd,ids,kind,separation,initial_check,witness=nothing)
    ccd_projected_separated(ccd,ids,kind,separation,ccd.refinement) && return UInt32(0)
    ccd_projected_moving_separated(ccd,ids,kind,separation,ccd.refinement) && return UInt32(0)
    # The queue already has an exact stored-endpoint stationary guard. Never
    # invoke the fused reference's narrowed common-translation shortcut.
    result=swept_primitive(ccd,ids,kind,separation,true)
    if result.hit && result.time<.5f0
        prefix=ccd_projected_prefix(ccd,ids,kind,separation,ccd.refinement)
        # As with Tight Inclusion's no_zero_toi time-window refinement, certify
        # only a shortened interval. Do not call that a resolved full query.
        prefix>result.time && (result=(;hit=true,time=prefix,resolved=false))
    end
    flags=UInt32(0)
    if result.hit
        bits=reinterpret(UInt32,.9f0*result.time)
        Atomix.@atomic min(ccd.bound[1],bits)
        record_ccd_primitive_bound!(witness,bits)
        result.resolved || (flags|=UInt32(1))
        result.time>0 || (flags|=UInt32(2))
        flags==0 || (Atomix.@atomic ccd.status[1]|=flags)
    end
    return flags
end

@inline function filtered_swept_triangle_pair!(ccd,a,b,separation,initial_check,witness=nothing)
    flags=UInt32(0)
    for k in Int32(1):Int32(3)
        a[k] in b || (flags|=filtered_swept_accumulate!(ccd,(a[k],b...),Val(true),separation,initial_check,witness))
        b[k] in a || (flags|=filtered_swept_accumulate!(ccd,(b[k],a...),Val(true),separation,initial_check,witness))
    end
    for i in Int32(1):Int32(3),j in Int32(1):Int32(3)
        aa,ab=a[i],a[i==3 ? Int32(1) : i+Int32(1)]
        ba,bb=b[j],b[j==3 ? Int32(1) : j+Int32(1)]
        (aa==ba || aa==bb || ab==ba || ab==bb) && continue
        flags|=filtered_swept_accumulate!(ccd,(aa,ab,ba,bb),Val(false),separation,initial_check,witness)
    end
    return flags
end
