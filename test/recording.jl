using PrintSim, Test
import PrintSim as PS
using GeometryBasics: Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

@testset "coupled recording preserves every kind of DOF" begin
    b = ModelBuilder(; gravity = Vec3f(0))
    axes = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))
    body = add_body!(TEST_DEVICE, b, box_mesh(Vec3(0.0, 0, -0.01), axes, (0.01, 0.01, 0.001)))
    add_stepper!(b, body)
    points, tets = tet_box((0.001, 0.001, 0.001), (2, 2, 2); origin = Vec3(0.1, 0.0, 0))
    add_deformable!(b, points, tets)
    add_sheet!(b, SheetGrid(0.01, 0.01, 5, 5), SheetMaterial())
    solver, state = WorldSolver(BACKEND, b; iterations = 8, newton = 2), WorldState(BACKEND, b)
    PS.prime!(solver, state)
    rec = Recording(b; names = ["shaft", "elastic part"])
    compact = Recording(b; names = ["shaft", "elastic part"], compact = true)
    record!(rec, solver, state, 0.0, "sample" => 2.0)
    record!(compact, solver, state, 0.0)
    PS.step!(solver, state, 1e-3)
    record!(rec, solver, state, 1e-3, "later sample" => 3.0)
    record!(compact, solver, state, 1e-3)
    @test isempty(compact.frames[1].surface)
    @test PS.frame_surface(compact, compact.frames[2]) ≈ rec.frames[2].surface atol = 1e-8
    @test collect(compact.frames[2].normal_force) == rec.frames[2].normal_force
    @test collect(compact.frames[2].sliding) == rec.frames[2].sliding
    @test length(rec.frames) == 2
    @test length(rec.frames[1].sheets) == 25
    @test length(rec.frames[1].bodies) == 4
    @test length(rec.frames[1].deformables) == length(b.deformables[1].x)
    @test length(rec.frames[1].surface) == 25 + sum(length(m.vertices) for m in b.meshes)
    @test rec.series["sample"][1] == 2 && isnan(rec.series["sample"][2])
    @test isnan(rec.series["later sample"][1]) && rec.series["later sample"][2] == 3
    @test all(isfinite, rec.frames[2].normal_force)
    mktemp() do path, io
        close(io)
        save_recording(path, rec)
        restored = load_recording(path)
        @test isequal(restored.series, rec.series)
        @test restored.frames[2].bodies == rec.frames[2].bodies
    end
    @test player(rec; series = ["sample"]) isa PS.Makie.Figure
    @test player(compact) isa PS.Makie.Figure
    mktempdir() do directory
        archive = Recording(b; compact = true, directory)
        record!(archive, solver, state, 1e-3)
        @test archive.frames isa PS.DiskFrames
        @test PS.recording_times(archive) == [1e-3]
        @test archive.frames[1].bodies == rec.frames[2].bodies
        @test isempty(archive.frames[1].strain)
        path = save_recording(joinpath(directory, "recording.jls"), archive)
        loaded = load_recording(path)
        @test PS.frame_surface(loaded, loaded.frames[1]) ≈ rec.frames[2].surface atol = 1e-8
        path=save_recording(joinpath(directory,"index.jls"),archive;external_geometry=true)
        index=open(PS.Serialization.deserialize,path)
        @test index isa PS.DiskRecordingIndex
        @test filesize(path)<16_384
        geometry_time=mtime(index.geometry)
        loaded=load_recording(path)
        @test PS.frame_surface(loaded,loaded.frames[1])≈rec.frames[2].surface atol=1e-8
        @test loaded.frames[1].deformables==archive.frames[1].deformables
        record!(archive,solver,state,2e-3)
        save_recording(path,archive;external_geometry=true)
        @test mtime(index.geometry)==geometry_time
        @test PS.recording_times(load_recording(path))==[1e-3,2e-3]
        # Resume an older observation while a later frame already exists.
        # The branch must append elsewhere, never overwrite the later evidence.
        old=PS.load_recording_data(index)
        branch=PS.branch_recording(old,joinpath(directory,"continuation"))
        @test branch.frames.directory!=archive.frames.directory
        @test branch.frames.files==old.frames.files
        record!(branch,solver,state,2e-3)
        @test branch.frames.files[end]!=archive.frames.files[end]
        @test length(old.frames)==1 && length(archive.frames)==2
        branch_path=save_recording(joinpath(directory,"branch.jls"),branch;external_geometry=true)
        @test PS.recording_times(load_recording(branch_path))==[1e-3,2e-3]
        @test PS.disk_recording_index(branch).geometry_digest==index.geometry_digest
        bad=PS.DiskRecordingIndex(1,index.geometry,repeat("0",64),index.frames,index.series)
        bad_path=joinpath(directory,"bad.jls")
        open(io->PS.Serialization.serialize(io,bad),bad_path,"w")
        @test_throws ArgumentError load_recording(bad_path)
    end
    @test_throws ArgumentError PS.disk_recording_index(rec)
end
