using PrintSim, Test
import PrintSim as PS
import Mantle
using GeometryBasics: Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

@testset "firmware command counters run entirely on the device" begin
    b = ModelBuilder(; gravity = Vec3f(0))
    axes = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))
    for x in (0.0, 0.1)
        body = add_body!(TEST_DEVICE, b, box_mesh(Vec3(x, 0, 0), axes, (0.01, 0.006, 0.006)))
        add_stepper!(b, body)
    end
    motors = PS.MotorModel(BACKEND, b)
    state = PS.MotorState(BACKEND, b)
    commands = Int32[2 -3; -5 4]
    control = MotorSequence(BACKEND, commands; rpm = 60 / 2048 / 0.001,
        waits = [0.002, 0.003, 0.002])
    graph = Mantle.Graph(Mantle.todevice(BACKEND))
    Mantle.repeat!(graph, 9) do _
        PS.launch!(graph, PS.motor_sequence!, (control, state), 1)
        PS.launch!(graph, PS.motor_clock!, (state, motors, 0.001f0), 2)
    end
    plan = Mantle.Plan(graph)
    Mantle.record!(plan)
    Mantle.run!(plan)
    @test Array(state.commanded) == [2, -5]
    @test Array(state.remaining) == [0, 0]
    @test Array(control.status)[3] == 0
    Mantle.run!(plan)
    @test Array(state.commanded) == [-1, -1]
    @test Array(state.remaining) == [0, 0]
    @test Array(control.status)[3] == 1
    Mantle.run!(plan)
    @test Array(state.commanded) == [-1, -1]
end
