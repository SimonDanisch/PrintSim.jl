using PrintSim,Test
import PrintSim as PS

@testset "checkpoint binding validation is complete before mutation" begin
    a=Float32[1,2]
    b=fill(PS.Position(PS.Vec3d(1,2,3)),2)
    roots=(;state=(;x=b,v=copy(b)),history=(;spring=a),control=(;status=Int32[1,2,3]))
    bindings=PS.checkpoint_array_bindings(roots)
    @test Set(keys(bindings))==Set([(:state,:x),(:state,:v),(:history,:spring),(:control,:status)])
    arrays=Dict(path=>copy(array) for (path,array) in bindings)
    @test PS.validate_checkpoint_arrays(bindings,arrays)===nothing
    @test PS.checkpoint_finite(b[1])
    @test !PS.checkpoint_finite(PS.Position(PS.Vec3d(NaN,0,0)))
    @test PS.checkpoint_digest(repeat("a",64))==repeat("a",64)
    @test_throws ArgumentError PS.checkpoint_digest("unfingerprinted")
    bad=copy(arrays); delete!(bad,(:state,:v))
    @test_throws ArgumentError PS.validate_checkpoint_arrays(bindings,bad)
    bad=copy(arrays); bad[(:history,:spring)]=Float64[1,2]
    @test_throws ArgumentError PS.validate_checkpoint_arrays(bindings,bad)
    bad=copy(arrays); bad[(:history,:spring)]=Float32[1,2,3]
    @test_throws ArgumentError PS.validate_checkpoint_arrays(bindings,bad)
    bad=copy(arrays); bad[(:history,:spring)]=Float32[NaN,2]
    @test_throws ArgumentError PS.validate_checkpoint_arrays(bindings,bad)
    @test a==Float32[1,2]
    @test b==roots.state.v
end

@testset "lossless sparse checkpoint history preserves shape and signed zero" begin
    original=zeros(Float32,40,30)
    original[3]=-0f0
    original[97]=2.5f0
    packed=PS.checkpoint_array_storage(original)
    @test packed isa PS.SparseCheckpointArray
    @test size(packed)==size(original)
    @test isequal(Array(packed),original)
    @test packed.indices==Int32[3,97]
    @test PS.validate_checkpoint_arrays(Dict((:x,)=>original),Dict((:x,)=>packed))===nothing
    @test PS.checkpoint_array_storage(ones(Float32,40,30)) isa Array
    unsorted=PS.SparseCheckpointArray(size(original),Int32[97,3],Float32[2.5,-0.])
    @test_throws ArgumentError PS.validate_checkpoint_arrays(Dict((:x,)=>original),Dict((:x,)=>unsorted))
    unsigned=PS.SparseCheckpointArray(size(original),UInt32[97,3],Float32[2.5,-0.])
    @test_throws ArgumentError PS.validate_sparse_checkpoint(unsigned)
    outside=PS.SparseCheckpointArray(size(original),Int32[1201],Float32[1])
    @test_throws ArgumentError PS.validate_sparse_checkpoint(outside)
    positions=fill(zero(PS.Position),1000)
    positions[10]=PS.Position(PS.Vec3d(.1,.2,.3))
    @test Array(PS.checkpoint_array_storage(positions))==positions
end
