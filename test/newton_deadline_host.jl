using PrintSim,Test
import PrintSim as PS

struct NewtonDeadlineProbe
    log::Vector{Symbol}
end
PS.Mantle.run!(p::NewtonDeadlineProbe)=(push!(p.log,:run);p)
PS.Mantle.waitfor!(p::NewtonDeadlineProbe)=(push!(p.log,:wait);p)

@testset "reusable stage deadline rejects expired work and preserves commit order" begin
    log=Symbol[]
    probe=NewtonDeadlineProbe(log)
    callback=i->push!(log,:commit)
    make(deadline)=PS.Stepper(PS.ReusableNewtonStepPlan(
        probe,probe,probe,2,1,callback,deadline,nothing,nothing),1,.001)
    @test_throws ErrorException PS.run!(make(0.))
    @test isempty(log)
    PS.run!(make(Inf))
    @test log==[:run,:wait,:run,:wait,:run,:wait,:run,:wait,:commit]
    empty!(log)
    converged=PS.Stepper(PS.ReusableNewtonStepPlan(
        probe,probe,probe,2,1,callback,Inf,UInt32[0,1],nothing),1,.001)
    PS.run!(converged)
    @test log==[:run,:wait,:run,:wait,:run,:wait,:commit]
    empty!(log)
    stale=PS.Stepper(PS.ReusableNewtonStepPlan(
        probe,probe,probe,2,1,callback,Inf,UInt32[0,0],nothing),1,.001)
    @test_throws ErrorException PS.run!(stale)
    @test :commit ∉ log
    empty!(log)
    rejected=PS.Stepper(PS.ReusableNewtonStepPlan(probe,probe,probe,2,1,callback,
        Inf,nothing,i->error("uncommitted diagnostic rejection")),1,.001)
    @test_throws ErrorException PS.run!(rejected)
    @test log==[:run,:wait,:run,:wait]
    empty!(log)
    split=PS.SplitNewtonStage(probe,probe,i->push!(log,:linear))
    @test_throws ErrorException PS.run_newton_stage!(split,1,0.)
    @test log==[:run,:wait,:linear]
    empty!(log)
    PS.run_newton_stage!(split,1,Inf)
    @test log==[:run,:wait,:linear,:run,:wait]
    empty!(log)
    rejected_linear=PS.SplitNewtonStage(probe,probe,i->error("reject linear candidate"))
    @test_throws ErrorException PS.run_newton_stage!(rejected_linear,1,Inf)
    @test log==[:run,:wait]
    empty!(log)
    prediction=PS.ObservedPredictionStage(probe,()->push!(log,:prediction))
    PS.run_prediction_stage!(prediction)
    @test log==[:run,:wait,:prediction]
end
