using PrintSim,Test,LinearAlgebra
import PrintSim as PS
isdefined(@__MODULE__,:BACKEND) || include("backend.jl")

function surface_contact_diagonal_fixture!(out,p)
    i=PS.KI.get_global_id(Int32).x
    i<=length(out) || return nothing
    @inbounds out[i]=(1f0+Float32(i)/32)*p[i]
    return nothing
end

@testset "device deep dyadic split-weight position and adjoint" begin
    bary=[(7001/8192,500/8192,691/8192),(27001/32768,2500/32768,3267/32768)]
    rows=fill(Int32.((1,2,3,4,5,6)),length(bary))
    weights=PS.quadratic_triangle_weights.(bary)
    host=PS.SurfaceEmbedding(Array,rows,weights,6)
    device=PS.SurfaceEmbedding(BACKEND,rows,weights,6)
    @test eltype(host.weight)===PS.SurfaceWeight
    p=PS.Vec3f[PS.Vec3f(sin(i),cos(i),sin(2i)) for i in 1:6]
    f=PS.Vec3f[PS.Vec3f(cos(i),sin(i),cos(2i)) for i in eachindex(bary)]
    forward=PS.upload(BACKEND,zeros(PS.Vec3f,length(bary)))
    transpose=PS.upload(BACKEND,zeros(PS.Vec3f,6))
    PS.launch!(BACKEND,PS.surface_map_forward!,(forward,PS.upload(BACKEND,p),device),length(bary))
    PS.launch!(BACKEND,PS.surface_map_transpose!,(transpose,PS.upload(BACKEND,f),device),6)
    @test Array(forward)≈[PS.surface_map_motion(host,i,p) for i in eachindex(bary)] atol=1e-6
    @test Array(transpose)≈[PS.surface_map_force(host,i,f) for i in 1:6] atol=1e-6
    @test sum(dot.(Array(forward),f))≈sum(dot.(p,Array(transpose))) atol=1e-6
    points=PS.Vec3d[PS.Vec3d(i/1000,i^2/10000,-i/100) for i in 1:6]
    x=PS.Position.(points)
    positions=PS.upload(BACKEND,zeros(PS.Position,length(bary)))
    PS.launch!(BACKEND,PS.surface_map_positions!,(positions,PS.upload(BACKEND,x),device,PS.upload(BACKEND,Float32[1])),length(bary))
    actual=[PS.Vec3d(v.hi)+PS.Vec3d(v.lo) for v in Array(positions)]
    reference=[sum(weights[i][k]*points[k] for k in 1:6) for i in eachindex(bary)]
    @test maximum(norm.(actual.-reference))<2e-14
end

@testset "device high-order surface forward and transpose maps" begin
    bary=[(i/8,j/8,(8-i-j)/8) for i in 0:8 for j in 0:8-i]
    rows=fill(Int32.((1,2,3,4,5,6)),length(bary))
    weights=PS.quadratic_triangle_weights.(bary)
    host=PS.SurfaceEmbedding(Array,rows,weights,6)
    device=PS.SurfaceEmbedding(BACKEND,rows,weights,6)
    p=PS.Vec3f[PS.Vec3f(sin(i),cos(i),sin(2i)) for i in 1:6]
    force=PS.Vec3f[PS.Vec3f(cos(i),sin(2i),cos(3i)) for i in eachindex(bary)]
    out=PS.upload(BACKEND,zeros(PS.Vec3f,length(bary)))
    transpose_out=PS.upload(BACKEND,zeros(PS.Vec3f,6))
    PS.launch!(BACKEND,PS.surface_map_forward!,(out,PS.upload(BACKEND,p),device),length(bary))
    PS.launch!(BACKEND,PS.surface_map_transpose!,(transpose_out,PS.upload(BACKEND,force),device),6)
    @test Array(out)≈[PS.surface_map_motion(host,i,p) for i in eachindex(bary)] atol=2e-6
    @test Array(transpose_out)≈[PS.surface_map_force(host,i,force) for i in 1:6] atol=2e-6
    @test sum(dot.(Array(out),force))≈sum(dot.(p,Array(transpose_out))) atol=3e-6
    coordinates=PS.Vec3d[(0,0,0),(1,0,0),(0,1,0),(.5,0,0),(0,.5,0),(.5,.5,0)]
    x=PS.Position.(coordinates .+ Ref(PS.Vec3d(.123,.234,.345)))
    positions=PS.upload(BACKEND,zeros(PS.Position,length(bary)))
    PS.launch!(BACKEND,PS.surface_map_positions!,(positions,PS.upload(BACKEND,x),device,PS.upload(BACKEND,Float32[1])),length(bary))
    actual=[PS.Vec3d(v.hi)+PS.Vec3d(v.lo) for v in Array(positions)]
    expected=[let v=PS.surface_map_position(host,i,x,1f0);PS.Vec3d(v.hi)+PS.Vec3d(v.lo);end for i in eachindex(bary)]
    @test maximum(norm.(actual.-expected))<2e-13
    workspace=PS.MappedSurfaceContact(BACKEND,device)
    fixed=UInt8[1,0,0,0,0,0]
    p[1]=zero(PS.Vec3f)
    result=PS.upload(BACKEND,fill(PS.Vec3f(1),6))
    PS.mapped_surface_product!(BACKEND,result,PS.upload(BACKEND,p),workspace,PS.upload(BACKEND,fixed),
        (target,out,direction)->PS.launch!(target,surface_contact_diagonal_fixture!,(out,direction),length(bary)))
    product=[(1f0+Float32(i)/32)*PS.surface_map_motion(host,i,p) for i in eachindex(bary)]
    expected_product=[PS.Vec3f(1)+(fixed[i]==0 ? PS.surface_map_force(host,i,product) : zero(PS.Vec3f)) for i in 1:6]
    @test Array(result)≈expected_product atol=1e-5
    blocks=fill(PS.Mat3f(1,0,0,0,1,0,0,0,1),length(bary))
    rhs=PS.upload(BACKEND,zeros(PS.Vec3f,6))
    inverse=PS.upload(BACKEND,fill(blocks[1],6))
    PS.launch!(BACKEND,PS.surface_map_rhs_add!,(rhs,inverse,PS.upload(BACKEND,force),PS.upload(BACKEND,blocks),device,PS.upload(BACKEND,fixed)),6)
    @test Array(rhs)≈[fixed[i]==0 ? PS.surface_map_force(host,i,force) : zero(PS.Vec3f) for i in 1:6] atol=1e-5
    @test Array(inverse)≈[fixed[i]==0 ? PS.preconditioner_inverse(blocks[1]+PS.surface_map_diagonal(host,i,blocks)) : blocks[1] for i in 1:6] atol=1e-6
end
