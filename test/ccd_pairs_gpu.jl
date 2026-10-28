using PrintSim,Test
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

@testset "pair-parallel CCD retains every broad-phase candidate" begin
    b=ModelBuilder()
    for z in (0.,0.,.1)
        add_sheet!(b,SheetGrid(.01,.01,2,2),SheetMaterial();place=p->PS.Vec3d(p[1],p[2],z))
    end
    reference=PS.SweptCollision(BACKEND,b)
    host=PS.ccd_snapshot(reference).data
    expected=Set{Tuple{Int32,Int32}}()
    for query in 1:host.query_count
        triangle=host.queries[query]
        own=host.owner[triangle]
        alo,ahi=PS.swept_face_box(host,host.faces[triangle])
        for other in eachindex(host.faces)
            other==triangle && continue
            owner=host.owner[other]
            host.dynamic[owner]!=0 && other<triangle && continue
            host.excluded[own,owner]!=0 && continue
            host.rigid[own]!=0 && host.rigid[own]==host.rigid[owner] && continue
            blo,bhi=PS.swept_face_box(host,host.faces[other])
            radius=host.radius[own]+host.radius[owner]
            all(ahi.+radius.>=blo) && all(alo.-radius.<=bhi) || continue
            push!(expected,(Int32(query),Int32(other)))
        end
    end
    n=host.query_count
    work=PS.swept_pair_workspace(BACKEND,n,128)
    PS.launch!(BACKEND,PS.swept_pair_candidates!,(work,reference,Int32(1),Int32(n),Val(false)),n)
    counts=Array(work.count)
    @test sum(counts)==length(expected)
    copyto!(work.offset,Int32.([0;cumsum(counts)[1:end-1]]))
    PS.launch!(BACKEND,PS.swept_pair_candidates!,(work,reference,Int32(1),Int32(n),Val(true)),n)
    @test only(Array(work.invalid))==0
    pairs=PS.ccd_host_range(work.pairs,1:sum(counts))
    @test Set(pairs)==expected
    PS.launch!(BACKEND,PS.swept_collisions!,(reference,true),n)
    result=PS.restore_ccd_snapshot(BACKEND,(;version=1,data=host))
    progress=NamedTuple[]
    processed=PS.initial_ccd_batches!(BACKEND,result;batch=n,pairwise=true,pair_batch=3,
        pair_capacity=128,on_batch=p->push!(progress,p))
    @test processed==n
    @test only(Array(result.status))==only(Array(reference.status))
    @test only(Array(result.bound))==only(Array(reference.bound))
    @test last(progress).candidate_pairs==length(expected)
    report=PS.ccd_failure_report(result)
    @test !isempty(report) && any(p->p.flags & UInt32(2)!=0,report)
    @test_throws ErrorException PS.initial_ccd_batches!(BACKEND,
        PS.restore_ccd_snapshot(BACKEND,(;version=1,data=host));batch=n,pairwise=true,pair_capacity=1)
    clear_builder=ModelBuilder()
    for z in (0.,.1)
        add_sheet!(clear_builder,SheetGrid(.01,.01,2,2),SheetMaterial();place=p->PS.Vec3d(p[1],p[2],z))
    end
    clear=PS.SweptCollision(BACKEND,clear_builder)
    @test PS.initial_ccd_batches!(BACKEND,clear;batch=3,pairwise=true,pair_batch=2,pair_capacity=128)==clear.query_count
    @test only(Array(clear.status)) & UInt32(2)==0
    limited=NamedTuple[]
    @test PS.initial_ccd_batches!(BACKEND,clear;batch=3,pairwise=true,pair_capacity=128,
        first_query=2,last_query=2,on_batch=p->push!(limited,p))==2
    @test length(limited)==1 && only(limited).first==only(limited).last==2
    @test_throws ArgumentError PS.initial_ccd_batches!(BACKEND,clear;first_query=3,last_query=1)
end
