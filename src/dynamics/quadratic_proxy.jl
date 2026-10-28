# High-Order IPC's parameter-space sampling and FE-basis displacement map.
# Unlike coordinate-tolerance welding, basis-row keys cannot merge nearby but
# topologically distinct printed surfaces. Adaptive edge splits are propagated
# across both incident triangles, so the proxy stays conforming.

"Canonical three-node key without allocating a vector during every sort comparison."
@inline function triangle_node_key(f)
    a,b,c=f
    a>b && ((a,b)=(b,a))
    b>c && ((b,c)=(c,b))
    a>b && ((a,b)=(b,a))
    return (a,b,c)
end

"Oriented P2 boundary faces of a T10 part, with their existing shared edge nodes."
function quadratic_boundary_faces(piece::TetPiece)
    linear=[ntuple(k->t[k],4) for t in piece.tets]
    boundary=tetrahedral_boundary(piece.x,linear)
    edge_nodes=Dict{Tuple{Int32,Int32},Int32}()
    for t in piece.tets,(k,(a,b)) in enumerate(((1,2),(1,3),(1,4),(2,3),(2,4),(3,4)))
        edge=minmax(t[a],t[b])
        node=t[k+4]
        get(edge_nodes,edge,node)==node || error("nonconforming quadratic edge nodes")
        edge_nodes[edge]=node
    end
    sort!(boundary;by=triangle_node_key)
    return [(a,b,c,edge_nodes[minmax(a,b)],edge_nodes[minmax(a,c)],edge_nodes[minmax(b,c)]) for (a,b,c) in boundary]
end

function quadratic_proxy_key(nodes,bary)
    values=quadratic_triangle_weights(bary)
    sum(values)==1 || error("proxy refinement exceeded exact dyadic basis precision")
    active=sort!([(Int32(nodes[k]),values[k]) for k in 1:6 if values[k]!=0];by=first)
    return ntuple(k->k<=length(active) ? active[k] : (Int32(0),0.),6)
end

function quadratic_proxy_point(key,x)
    anchor=x[key[1][1]]
    value=anchor
    for k in 2:6
        node,weight=key[k]
        node==0 || (value+=Float64(weight)*(x[node]-anchor))
    end
    return value
end

@inline function quadratic_patch_tangents(nodes,bary,x)
    a,b,c=bary
    du=(-(4a-1),4b-1,zero(a),4(a-b),-4c,4c)
    dv=(-(4a-1),zero(a),4c-1,-4b,4(a-c),4b)
    anchor=x[nodes[1]]
    u,v=zero(anchor),zero(anchor)
    for k in 2:6
        delta=x[nodes[k]]-anchor
        u+=du[k]*delta
        v+=dv[k]*delta
    end
    return u,v
end

"Exact host fallback for an uncertain orientation filter; never moves vertices."
function quadratic_proxy_oriented_exact(nodes,uv,face,points,x)
    Q=Rational{BigInt}
    exact(v)=Vec3{Q}(Q(v[1]),Q(v[2]),Q(v[3]))
    a,b,c=map(i->exact(points[i]),face)
    normal=cross(b-a,c-a)
    all(iszero,normal) && return false
    local_x=map(i->exact(x[i]),nodes)
    parameters=map(exact,uv)
    half=Q(1,2)
    samples=(parameters...,(parameters[1]+parameters[2])*half,
        (parameters[2]+parameters[3])*half,(parameters[3]+parameters[1])*half)
    values=map(samples) do bary
        u,v=quadratic_patch_tangents((1,2,3,4,5,6),bary,local_x)
        dot(normal,cross(u,v))
    end
    all(>(0),values[1:3]) || return false
    return all(k->2values[k+3]-(values[k]+values[k==3 ? 1 : k+1])/2>0,1:3)
end

"Local orientation refinement criterion; global intersection auditing is still required."
function quadratic_proxy_oriented(nodes,uv,face,points,x)
    a,b,c=map(i->points[i],face)
    e1,e2=b-a,c-a
    normal=cross(e1,e2)
    scale=norm(normal)
    scale>0 || return quadratic_proxy_oriented_exact(nodes,uv,face,points,x)
    normal_error=16eps(Float64)*norm(e1)*norm(e2)
    # Bound derivative arithmetic before cancellation, not from the possibly
    # tiny final tangents. P2 derivative coefficients have magnitude <=4 on
    # the reference triangle. Uncertain cases use exact rational arithmetic.
    anchor=x[nodes[1]]
    tangent_bound=4sum(k->norm(x[nodes[k]]-anchor),2:6)
    jacobian_bound=tangent_bound^2
    samples=(uv...,(uv[1]+uv[2])/2,(uv[2]+uv[3])/2,(uv[3]+uv[1])/2)
    values=ntuple(6) do k
        u,v=quadratic_patch_tangents(nodes,samples[k],x)
        value=dot(normal,cross(u,v))
        error=(512eps(Float64)*scale+normal_error)*jacobian_bound
        (value,error)
    end
    # normal·(du×dv) is quadratic in parameter space. Its six triangular
    # Bernstein coefficients bound it across the WHOLE facet, not just the
    # centroid. Mid-edge coefficient = 2f(mid) - (f(a)+f(b))/2.
    for k in 1:3
        value,error=values[k]
        value < -error && return false
        value > error || return quadratic_proxy_oriented_exact(nodes,uv,face,points,x)
    end
    for k in 1:3
        j=k==3 ? 1 : k+1
        middle,em=values[k+3]
        left,el=values[k]
        right,er=values[j]
        coefficient=2middle-(left+right)/2
        error=2em+(el+er)/2+16eps(Float64)*(2abs(middle)+(abs(left)+abs(right))/2)
        coefficient < -error && return false
        coefficient>error || return quadratic_proxy_oriented_exact(nodes,uv,face,points,x)
    end
    return true
end

"""
    quadratic_surface_proxy(rest, face_nodes; initial=rest, tolerance=1e-9,
                            max_faces=2_000_000, max_rounds=10)

Build a conforming P1 collision proxy of P2 triangular FE faces (node order
a,b,c,ab,ac,bc), retaining exact reference coordinates and the sparse FE map.
The quadratic patch/linear-facet deviation is bounded by 4/3 times the largest
edge-midpoint deviation. The bound applies to `rest` and `initial`, not future states;
facets also refine until their normals agree with both P2 patches throughout
their parameter triangle. The coupled solver must monitor deformation. This constructor
does not certify global intersection-freedom or replace the collision audit.
`minimum_rounds=0` permits unrefined flat patches when the same orientation and
error tests pass; the default retains one initial subdivision. Neither choice
certifies a future deformation without the runtime proxy guard.
"""
function quadratic_surface_proxy(rest,face_nodes;initial=rest,tolerance=1e-9,
        max_faces=2_000_000,max_rounds=10,minimum_rounds=1,on_round=nothing,on_failure=nothing)
    length(rest)==length(initial) || throw(ArgumentError("proxy state size mismatch"))
    isfinite(tolerance) && tolerance>0 && max_faces>0 && 1<=max_rounds<=40 ||
        throw(ArgumentError("invalid proxy accuracy/resource bounds"))
    minimum_rounds isa Integer && 0<=minimum_rounds<=max_rounds ||
        throw(ArgumentError("invalid minimum proxy refinement rounds"))
    length(face_nodes)<=max_faces || error("proxy input already exceeds face budget")
    all(f->length(f)==6 && length(unique(f))==6 && all(i->1<=i<=length(rest),f),face_nodes) ||
        throw(ArgumentError("proxy faces need six distinct FE nodes"))
    Key=NTuple{6,Tuple{Int32,Float64}}
    keys=Key[]
    lookup=Dict{Key,Int32}()
    positions=Vec3{Float64}[]
    reference=Vec3{Float64}[]
    function vertex(parent,bary)
        key=quadratic_proxy_key(face_nodes[parent],bary)
        return get!(lookup,key) do
            length(keys)<typemax(Int32) || error("proxy exceeds Int32 vertices")
            push!(keys,key)
            push!(positions,quadratic_proxy_point(key,initial))
            push!(reference,quadratic_proxy_point(key,rest))
            Int32(length(keys))
        end
    end
    faces=NTuple{3,Int32}[]
    parent=Int32[]
    parameters=NTuple{3,Vec3{Float64}}[]
    corners=(Vec3{Float64}(1,0,0),Vec3{Float64}(0,1,0),Vec3{Float64}(0,0,1))
    for p in eachindex(face_nodes)
        push!(faces,ntuple(k->vertex(p,corners[k]),3))
        push!(parent,Int32(p));push!(parameters,corners)
    end
    max_error=0.0
    for round in 0:max_rounds
        marked=Set{Tuple{Int32,Int32}}()
        max_error=0.0
        orientation_refinements=0
        for t in eachindex(faces)
            f,uv,p=faces[t],parameters[t],parent[t]
            orientation_failed=round>=minimum_rounds && !(quadratic_proxy_oriented(face_nodes[p],uv,f,positions,initial) &&
                    quadratic_proxy_oriented(face_nodes[p],uv,f,reference,rest))
            if orientation_failed
                orientation_refinements+=1
                # Longest-edge bisection improves a skinny facet's aspect ratio.
                # Splitting all three edges keeps its bad aspect ratio forever,
                # eventually shrinking the thin direction below position precision.
                longest=1
                longest_length=0.
                for k in 1:3
                    j=k==3 ? 1 : k+1
                    edge_length=norm(reference[f[k]]-reference[f[j]])
                    if edge_length>longest_length
                        longest,longest_length=k,edge_length
                    end
                end
                push!(marked,minmax(f[longest],f[longest==3 ? 1 : longest+1]))
            end
            worst_edge=1
            worst_bound=-Inf
            for k in 1:3
                j=k==3 ? 1 : k+1
                midpoint=(uv[k]+uv[j])/2
                key=quadratic_proxy_key(face_nodes[p],midpoint)
                bound=0.0
                for (state,proxy) in ((initial,positions),(rest,reference))
                    exact=quadratic_proxy_point(key,state)
                    a,b=proxy[f[k]],proxy[f[j]]
                    deviation=norm(exact-(a+b)/2)
                    roundoff=64eps(Float64)*max(maximum(abs,exact),maximum(abs,a),maximum(abs,b),floatmin(Float64))
                    bound=max(bound,(4/3)*(deviation+roundoff)*(1+16eps(Float64)))
                end
                max_error=max(max_error,bound)
                if bound>worst_bound
                    worst_edge,worst_bound=k,bound
                end
                round<minimum_rounds && push!(marked,minmax(f[k],f[j]))
            end
            # Bisect in the approximation-error metric, not isotropically.
            # Rank-one curvature (a bent flat leaf) needs strips along the
            # bend, not equally fine sampling across its uncurved width.
            if round>=minimum_rounds && worst_bound>tolerance
                push!(marked,minmax(f[worst_edge],f[worst_edge==3 ? 1 : worst_edge+1]))
            end
        end
        on_round===nothing || on_round((;round,faces=length(faces),vertices=length(keys),marked_edges=length(marked),max_error,tolerance,orientation_refinements))
        isempty(marked) && return (;points=reference,initial=positions,faces,parent,parameters,
            nodes=[ntuple(k->key[k][1],6) for key in keys],
            weights=[ntuple(k->key[k][2],6) for key in keys],max_error,tolerance,rounds=round,oriented_facets=true)
        if round==max_rounds
            # Preserve diagnostic evidence without returning an unvalidated proxy.
            on_failure===nothing || on_failure((;points=reference,initial=positions,faces,parent,parameters,marked,max_error,round))
            error("quadratic proxy did not meet tolerance within refinement budget (error=$max_error, faces=$(length(faces)))")
        end
        if orientation_refinements>0
            # Longest-edge propagation closure: a neighbour must not be split
            # solely through a short edge. Repeated arbitrary green refinements
            # otherwise create increasingly thin children. Propagate each
            # marked edge to the longest reference edge of both incident faces.
            Edge=Tuple{Int32,Int32}
            incident_longest=Dict{Edge,Tuple{Edge,Edge}}()
            empty_edge=(Int32(0),Int32(0))
            for f in faces
                edges=ntuple(k->minmax(f[k],f[k==3 ? 1 : k+1]),3)
                lengths=map(e->sum(abs2,reference[e[1]]-reference[e[2]]),edges)
                longest=edges[argmax(lengths)]
                for e in edges
                    previous=get(incident_longest,e,(empty_edge,empty_edge))
                    previous[2]==empty_edge || error("nonmanifold proxy edge during refinement")
                    incident_longest[e]=(longest,previous[1])
                end
            end
            queue=collect(marked)
            index=1
            while index<=length(queue)
                edge=queue[index]
                index+=1
                for longest in incident_longest[edge]
                    if longest!=empty_edge && !(longest in marked)
                        push!(marked,longest)
                        push!(queue,longest)
                    end
                end
            end
        end
        target_faces=sum(f->1+sum(k->minmax(f[k],f[k==3 ? 1 : k+1]) in marked,1:3),faces;init=0)
        target_faces<=max_faces || error("quadratic proxy refinement exceeds face budget ($target_faces > $max_faces)")
        next_faces=NTuple{3,Int32}[]
        next_parent=Int32[]
        next_parameters=NTuple{3,Vec3{Float64}}[]
        sizehint!(next_faces,target_faces);sizehint!(next_parent,target_faces);sizehint!(next_parameters,target_faces)
        function emit(f,uv,p)
            push!(next_faces,f);push!(next_parameters,uv);push!(next_parent,p)
        end
        for t in eachindex(faces)
            f,uv,p=faces[t],parameters[t],parent[t]
            split=ntuple(k->minmax(f[k],f[k==3 ? 1 : k+1]) in marked,3)
            count=sum(split)
            if count==0
                emit(f,uv,p)
                continue
            end
            # Rotate cyclically to the one-edge or two-edge template.
            start=count==1 ? something(findfirst(identity,split),1) : count==2 ? mod1(something(findfirst(!,split),1)+1,3) : 1
            order=ntuple(k->mod1(start+k-1,3),3)
            a,b,c=map(k->f[k],order)
            ua,ub,uc=map(k->uv[k],order)
            uab=(ua+ub)/2
            ab=vertex(p,uab)
            if count==1
                emit((a,ab,c),(ua,uab,uc),p)
                emit((ab,b,c),(uab,ub,uc),p)
            else
                ubc=(ub+uc)/2
                bc=vertex(p,ubc)
                if count==2
                    emit((b,bc,ab),(ub,ubc,uab),p)
                    # The green quadrilateral admits two diagonals. A fixed
                    # diagonal can keep introducing reversed chords on a thin
                    # curved patch while its neighbours refine indefinitely.
                    # Prefer the triangulation with more certified facets;
                    # break ties by the shorter reference-space diagonal.
                    first_faces=((a,ab,c),(ab,bc,c))
                    first_uv=((ua,uab,uc),(uab,ubc,uc))
                    second_faces=((a,ab,bc),(a,bc,c))
                    second_uv=((ua,uab,ubc),(ua,ubc,uc))
                    score_first=sum(k->quadratic_proxy_oriented(face_nodes[p],first_uv[k],first_faces[k],positions,initial) &&
                        quadratic_proxy_oriented(face_nodes[p],first_uv[k],first_faces[k],reference,rest),1:2)
                    score_second=sum(k->quadratic_proxy_oriented(face_nodes[p],second_uv[k],second_faces[k],positions,initial) &&
                        quadratic_proxy_oriented(face_nodes[p],second_uv[k],second_faces[k],reference,rest),1:2)
                    use_second=score_second>score_first || (score_second==score_first &&
                        norm(reference[a]-reference[bc])<norm(reference[ab]-reference[c]))
                    chosen_faces,chosen_uv=use_second ? (second_faces,second_uv) : (first_faces,first_uv)
                    for k in 1:2
                        emit(chosen_faces[k],chosen_uv[k],p)
                    end
                else
                    uca=(uc+ua)/2
                    ca=vertex(p,uca)
                    emit((a,ab,ca),(ua,uab,uca),p)
                    emit((ab,b,bc),(uab,ub,ubc),p)
                    emit((ca,bc,c),(uca,ubc,uc),p)
                    emit((ab,bc,ca),(uab,ubc,uca),p)
                end
            end
        end
        faces,parent,parameters=next_faces,next_parent,next_parameters
    end
    error("unreachable proxy refinement state")
end
