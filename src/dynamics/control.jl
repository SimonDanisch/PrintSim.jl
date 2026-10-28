# Motor commands are control inputs, never prescribed mechanical poses. A sequence waits
# for every motor's step counter, not for the actual rotor to catch up: just like firmware.

struct MotorSequence{I, F}
    steps::I
    rpm::F
    waits::I
    status::I                 # next command (1-based), wait ticks, finished flag
    commands::Int
    motors::Int
    timestep::Float32
end
Adapt.@adapt_structure MotorSequence

"""
    MotorSequence(backend, steps; rpm=9.75, waits=zeros(ncommands+1), timestep=1e-3)

Device-side commands, one column per move and one row per motor; steps are signed.
`waits` gives the idle time before each move and after the last. Each motor switches off
at its own last step. The next move starts only after all counters finish, even if a
loaded motor has mechanically lost steps. Waiting and command advancement need no readback.
"""
function MotorSequence(backend, steps::AbstractMatrix{<:Integer}; rpm = 9.75,
        waits = zeros(size(steps, 2) + 1), timestep = 1e-3)
    n, commands = size(steps)
    n > 0 && commands > 0 || throw(ArgumentError("a motor sequence needs motors and commands"))
    timestep > 0 && isfinite(timestep) || throw(ArgumentError("timestep must be positive and finite"))
    length(waits) == commands + 1 && all(x -> isfinite(x) && x >= 0, waits) ||
        throw(ArgumentError("waits must have one nonnegative duration before each command and after the last"))
    speed = rpm isa Number ? fill(Float32(rpm), size(steps)) : Float32.(rpm)
    size(speed) == size(steps) && all(x -> isfinite(x) && x > 0, speed) ||
        throw(ArgumentError("rpm must be positive, finite and match the command matrix"))
    ticks = Int32[ceil(Int, w / timestep) for w in waits]
    up(a) = upload(backend, a)
    return MotorSequence(up(Int32.(steps)), up(speed), up(reshape(ticks, :, 1)),
        up(reshape(Int32[1, ticks[1], 0], :, 1)), commands, n, Float32(timestep))
end

function motor_sequence!(control, state)
    KI.get_global_id(Int32).x == 1 || return nothing
    @inbounds begin
        control.status[3] != 0 && return nothing
        for motor in Int32(1):Int32(control.motors)
            state.remaining[motor] > 0 && return nothing
        end
        if control.status[2] > 0
            control.status[2] -= Int32(1)
            return nothing
        end
        command = control.status[1]
        if command > control.commands
            control.status[3] = 1
            return nothing
        end
        for motor in Int32(1):Int32(control.motors)
            steps = control.steps[motor, command]
            state.remaining[motor] = abs(steps)
            state.direction[motor] = steps >= 0 ? Int32(1) : -Int32(1)
            state.rpm[motor] = control.rpm[motor, command]
            state.fraction[motor] = 0
        end
        control.status[1] = command + Int32(1)
        control.status[2] = control.waits[command + Int32(1)]
    end
    return nothing
end

motor_solver(s::WorldSolver) = s.bodies
motor_solver(s::BodySolver) = s

function Stepper(s::Union{BodySolver, FEMSolver}, state, dt, steps; control = nothing)
    steps > 0 && isfinite(dt) && dt > 0 || throw(ArgumentError("positive step count and timestep required"))
    if control !== nothing
        s isa FEMSolver && throw(ArgumentError("a finite-element-only solver has no motors"))
        control.motors == motor_solver(s).motors.motors || throw(ArgumentError("motor count mismatch"))
        control.timestep == Float32(dt) || throw(ArgumentError("control timestep mismatch"))
    end
    g = Mantle.Graph(Mantle.todevice(s.backend))
    Mantle.repeat!(g, steps) do _
        control === nothing || launch!(g, motor_sequence!, (control, motor_solver(s).motor_state), 1)
        step!(g, s, state, dt)
    end
    plan = Mantle.Plan(g)
    Mantle.record!(plan)
    return Stepper(plan, steps, Float64(dt))
end

# Mantle loops do not nest. A coupled timestep contains a device CG loop, so
# batches replay the same timestep plan rather than expanding CG on the host.
# Motor commands and convergence decisions still execute entirely on the GPU.
struct WorldStepPlan{P, F}
    plan::P
    repetitions::Int
    on_step::F
end

# The Newton stage names only persistent world/state buffers. Recording it once
# bounds command memory independently of the requested Newton iteration count.
# This is host orchestration of resident GPU kernels, not a host physics solve.
struct ReusableNewtonStepPlan{P,N,F,A,C,X}
    predict::P
    newton::N
    finish::F
    iterations::Int
    repetitions::Int
    on_step::A
    deadline::Float64
    convergence::C
    on_newton::X
end

# Split only at a synchronization-safe boundary: all Newton/PCG state and
# contact-bounded candidate positions are resident before CCD/line search.
# This also permits refusing the expensive trial before a work deadline.
struct SplitNewtonStage{L,T,A}
    linear::L
    trial::T
    on_linear::A
end
struct ObservedPredictionStage{P,A}
    plan::P
    on_prediction::A
end

"""Resident execution cursor for a reusable Newton plan.

Only synchronized stage boundaries may pause. A partial timestep is never
finalized, observed or checkpointed. This cursor is NOT a cold checkpoint: a
device loss, source change or process retirement invalidates its resident
buffers. Any execution/audit exception poisons it and forbids retry.
"""
mutable struct ReusableNewtonCursor{S}
    stepper::S
    step::Int
    iteration::Int
    phase::Symbol
    poisoned::Bool
end

function ReusableNewtonCursor(st::Stepper{<:ReusableNewtonStepPlan})
    return ReusableNewtonCursor(st,1,1,:predict,false)
end

function reusable_cursor_receipt(cursor,stages)
    return (;scope=:resident_newton_execution_not_cold_checkpoint,
        complete=cursor.phase===:done,completed_steps=cursor.step-1,
        next_step=cursor.step,next_iteration=cursor.iteration,
        next_phase=cursor.phase,executed_stages=stages)
end

"Advance synchronized resident stages within a new bounded reservation."
function advance_reusable_newton!(cursor::ReusableNewtonCursor;deadline,max_stages=typemax(Int),stage_margin=0.)
    cursor.poisoned && error("resident Newton cursor is poisoned; restore an accepted checkpoint")
    isfinite(deadline) && deadline>0 || throw(ArgumentError("finite absolute stage deadline required"))
    max_stages isa Integer && max_stages>0 || throw(ArgumentError("positive stage budget required"))
    isfinite(stage_margin) && stage_margin>=0 || throw(ArgumentError("nonnegative finite stage safety margin required"))
    p=cursor.stepper.plan
    stages=0
    while cursor.phase!==:done && stages<max_stages && time()+stage_margin<deadline
        try
            if cursor.phase===:predict
                run_prediction_stage!(p.predict)
                cursor.phase=p.newton isa SplitNewtonStage ? :linear : :trial
            elseif cursor.phase===:linear
                stage=p.newton
                Mantle.run!(stage.linear);Mantle.waitfor!(stage.linear)
                stage.on_linear===nothing || stage.on_linear(cursor.iteration)
                cursor.phase=:trial
            elseif cursor.phase===:trial
                trial=p.newton isa SplitNewtonStage ? p.newton.trial : p.newton
                Mantle.run!(trial);Mantle.waitfor!(trial)
                p.on_newton===nothing || p.on_newton(cursor.iteration)
                converged=false
                if p.convergence!==nothing
                    status=Array(p.convergence)
                    status[2]==UInt32(cursor.iteration) && status[1] in (UInt32(0),UInt32(1)) ||
                        error("invalid resident Newton execution receipt; timestep not committed")
                    converged=status[1]==0
                end
                if converged || cursor.iteration==p.iterations
                    cursor.phase=:finish
                else
                    cursor.iteration+=1
                    cursor.phase=p.newton isa SplitNewtonStage ? :linear : :trial
                end
            elseif cursor.phase===:finish
                Mantle.run!(p.finish);Mantle.waitfor!(p.finish)
                # Acceptance remains exclusively in the existing audit callback.
                p.on_step===nothing || p.on_step(cursor.step)
                cursor.step+=1
                cursor.iteration=1
                cursor.phase=cursor.step>p.repetitions ? :done : :predict
            else
                error("invalid resident Newton cursor phase")
            end
            stages+=1
        catch
            cursor.poisoned=true
            rethrow()
        end
    end
    return reusable_cursor_receipt(cursor,stages)
end
run_prediction_stage!(stage)=(Mantle.run!(stage);Mantle.waitfor!(stage);nothing)
function run_prediction_stage!(stage::ObservedPredictionStage)
    Mantle.run!(stage.plan);Mantle.waitfor!(stage.plan)
    stage.on_prediction()
    return nothing
end
function run_newton_stage!(stage,iteration,deadline)
    Mantle.run!(stage);Mantle.waitfor!(stage)
    return nothing
end
function run_newton_stage!(stage::SplitNewtonStage,iteration,deadline)
    Mantle.run!(stage.linear);Mantle.waitfor!(stage.linear)
    stage.on_linear===nothing || stage.on_linear(iteration)
    time()<deadline || error("reusable timestep deadline reached after linear solve; trial not launched, timestep not committed")
    Mantle.run!(stage.trial);Mantle.waitfor!(stage.trial)
    return nothing
end

function check_world_recording_size(parts::Integer;limit=4096)
    limit isa Integer && limit>0 && parts>=0 ||
        throw(ArgumentError("positive world recording limit and nonnegative piece count required"))
    parts<=limit || throw(ArgumentError(
        "world plan needs $parts command-buffer pieces (limit $limit); " *
        "use reusable_newton=true to avoid retaining every expanded Newton stage. " *
        "Recording was refused before opening any command buffers."))
    return nothing
end

function record_world_plan!(plan;limit)
    check_world_recording_size(length(Mantle.partitionranges(plan));limit)
    Mantle.record!(plan)
    return plan
end

function Stepper(s::WorldSolver, state::WorldState, dt, steps; control = nothing, solve_history = nothing,
        on_step = nothing,on_prepared=nothing,policy=Mantle.InOrder(),reusable_newton=false,
        recording_limit=4096,stage_deadline=Inf,compact_linesearch=true,on_newton=nothing,
        split_newton=false,on_linear=nothing,on_prediction=nothing)
    steps > 0 && isfinite(dt) && dt > 0 || throw(ArgumentError("positive step count and timestep required"))
    check_world_recording_size(0;limit=recording_limit)
    on_newton===nothing || reusable_newton ||
        throw(ArgumentError("per-Newton callbacks require a reusable Newton plan"))
    (!split_newton || reusable_newton) && (on_linear===nothing || split_newton) ||
        throw(ArgumentError("split Newton and linear callbacks require reusable_newton=true and split_newton=true"))
    on_prediction===nothing || reusable_newton ||
        throw(ArgumentError("prediction callbacks require reusable_newton=true"))
    stage_deadline isa Real && !isnan(stage_deadline) ||
        throw(ArgumentError("stage deadline must be a real absolute time"))
    if control !== nothing
        control.motors == s.bodies.motors.motors || throw(ArgumentError("motor count mismatch"))
        control.timestep == Float32(dt) || throw(ArgumentError("control timestep mismatch"))
    end
    if reusable_newton
        phases=split_newton ? (:predict,:newton_linear,:newton_trial,:finish) : (:predict,:newton,:finish)
        plans=map(phases) do phase
            graph=Mantle.Graph(Mantle.todevice(s.backend))
            if phase===:predict
                control===nothing || launch!(graph,motor_sequence!,(control,s.bodies.motor_state),1)
            end
            step!(graph,s,state,dt;solve_history,phase,newton_phases=1:1,
                dynamic_newton=phase in (:newton,:newton_linear,:newton_trial),compact_linesearch)
            on_prepared===nothing || on_prepared(Symbol(phase,:_graph_declared))
            plan=Mantle.Plan(graph;policy)
            record_world_plan!(plan;limit=recording_limit)
            on_prepared===nothing || on_prepared(Symbol(phase,:_plan_ready))
            plan
        end
        convergence=s.newton_tolerance>0 ? s.energy.newton_status : nothing
        stage=split_newton ? SplitNewtonStage(plans[2],plans[3],on_linear) : plans[2]
        prediction=on_prediction===nothing ? plans[1] : ObservedPredictionStage(plans[1],on_prediction)
        return Stepper(ReusableNewtonStepPlan(prediction,stage,plans[end],s.newton,steps,on_step,
            Float64(stage_deadline),convergence,on_newton),steps,Float64(dt))
    end
    g = Mantle.Graph(Mantle.todevice(s.backend))
    control === nothing || launch!(g, motor_sequence!, (control, s.bodies.motor_state), 1)
    step!(g, s, state, dt; solve_history,compact_linesearch)
    on_prepared===nothing || on_prepared(:timestep_graph_declared)
    plan = Mantle.Plan(g;policy)
    on_prepared===nothing || on_prepared(:timestep_graph_compiled)
    record_world_plan!(plan;limit=recording_limit)
    on_prepared===nothing || on_prepared(:expanded_plan_ready)
    return Stepper(WorldStepPlan(plan, steps, on_step), steps, Float64(dt))
end

function run!(st::Stepper{<:WorldStepPlan})
    for step in 1:st.plan.repetitions
        # A completed first run supplies timings that may rebuild the recording.
        # Keep its command buffers out of flight before that adaptive rebuild.
        step > 1 && Mantle.waitfor!(st.plan.plan)
        Mantle.run!(st.plan.plan)
        if st.plan.on_step !== nothing
            Mantle.waitfor!(st.plan.plan)
            st.plan.on_step(step)
        end
    end
    return st
end

waitfor!(st::Stepper{<:WorldStepPlan}) = (Mantle.waitfor!(st.plan.plan); st)

function run!(st::Stepper{<:ReusableNewtonStepPlan})
    p=st.plan
    for step in 1:p.repetitions
        time()<p.deadline || error("reusable timestep deadline reached before prediction; timestep not committed")
        run_prediction_stage!(p.predict)
        for iteration in 1:p.iterations
            time()<p.deadline || error("reusable timestep deadline reached before Newton stage; timestep not committed")
            run_newton_stage!(p.newton,iteration,p.deadline)
            p.on_newton===nothing || p.on_newton(iteration)
            if p.convergence!==nothing
                # The GPU computes convergence. Read only its two-word receipt
                # after this already synchronized stage, avoiding submissions
                # of long inactive CCD schedules on every remaining Newton.
                status=Array(p.convergence)
                status[2]==UInt32(iteration) && status[1] in (UInt32(0),UInt32(1)) ||
                    error("invalid reusable Newton execution receipt; timestep not committed")
                status[1]==0 && break
            end
        end
        time()<p.deadline || error("reusable timestep deadline reached before finalization; timestep not committed")
        Mantle.run!(p.finish);Mantle.waitfor!(p.finish)
        p.on_step===nothing || p.on_step(step)
    end
    return st
end
waitfor!(st::Stepper{<:ReusableNewtonStepPlan})=(Mantle.waitfor!(st.plan.finish);st)
