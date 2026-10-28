using PrintSim,Test
import PrintSim as PS

struct WorldRecordingGuardProbe
    parts::Int
    opened::Base.RefValue{Int}
end
PS.Mantle.partitionranges(p::WorldRecordingGuardProbe)=fill(1:1,p.parts)
PS.Mantle.record!(p::WorldRecordingGuardProbe)=(p.opened[]+=1;p)

@testset "world recording expansion fails before driver allocation" begin
    @test PS.check_world_recording_size(4096)===nothing
    @test PS.check_world_recording_size(0)===nothing
    @test_throws ArgumentError PS.check_world_recording_size(4097)
    @test_throws ArgumentError PS.check_world_recording_size(1;limit=0)
    @test_throws ArgumentError PS.check_world_recording_size(-1)
    @test_throws ArgumentError PS.check_world_recording_size(1;limit=1.5)
    @test PS.check_world_recording_size(3;limit=3)===nothing
    rejected=WorldRecordingGuardProbe(4097,Ref(0))
    @test_throws ArgumentError PS.record_world_plan!(rejected;limit=4096)
    @test rejected.opened[]==0
    accepted=WorldRecordingGuardProbe(3,Ref(0))
    @test PS.record_world_plan!(accepted;limit=3)===accepted
    @test accepted.opened[]==1
end
