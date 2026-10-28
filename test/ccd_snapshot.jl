using PrintSim,Test,Serialization
import PrintSim as PS

@testset "collision-only snapshots retain input and reject incompatible data" begin
    b=ModelBuilder()
    add_sheet!(b,SheetGrid(.01,.01,2,2),SheetMaterial())
    ccd=PS.SweptCollision(Array,b)
    snapshot=PS.ccd_snapshot(ccd)
    restored=PS.restore_ccd_snapshot(Array,snapshot)
    for name in fieldnames(typeof(ccd))
        @test getfield(restored,name)==getfield(ccd,name)
    end
    @test restored.start !== ccd.start
    @test PS.initial_ccd_batches!(Array,ccd;max_seconds=0)==0
    @test PS.initial_ccd_batches!(Array,ccd;first_query=2,max_seconds=0)==1
    @test_throws ArgumentError PS.initial_ccd_batches!(Array,ccd;first_query=0,max_seconds=0)
    @test_throws ArgumentError PS.initial_ccd_batches!(Array,ccd;max_seconds=-1)
    @test_throws ArgumentError PS.restore_ccd_snapshot(Array,merge(snapshot,(version=2,)))
    malformed=merge(snapshot.data,(blocking=Int32[],))
    @test_throws ArgumentError PS.restore_ccd_snapshot(Array,(;version=1,data=malformed))
    mktempdir() do directory
        path=joinpath(directory,"initial_ccd.jls")
        @test PS.save_ccd_snapshot(path,ccd;metadata=(purpose=:test,))==path
        saved=open(deserialize,path)
        @test saved.metadata.purpose==:test
        @test saved.data.faces==ccd.faces
        @test_throws ArgumentError PS.save_ccd_snapshot(path,ccd)
    end
end
