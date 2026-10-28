using PrintSim,Test
import PrintSim as PS

struct CursorTestPlan{F}
    action::F
end
PS.Mantle.run!(plan::CursorTestPlan)=(plan.action();plan)
PS.Mantle.waitfor!(plan::CursorTestPlan)=plan

function cursor_test_stepper(;split=true,repetitions=2,bad_receipt=false,fail_audit=false)
    events=Symbol[]
    status=UInt32[1,0]
    predict=CursorTestPlan(()->begin push!(events,:predict);status[:]=UInt32[1,0];end)
    linear=CursorTestPlan(()->push!(events,:linear))
    trial=CursorTestPlan(()->begin
        push!(events,:trial)
        status[2]+=1
        status[1]=bad_receipt ? 7 : status[2]==2 ? 0 : 1
    end)
    finish=CursorTestPlan(()->push!(events,:finish))
    stage=split ? PS.SplitNewtonStage(linear,trial,i->push!(events,:linear_audit)) : trial
    stepper=Stepper(PS.ReusableNewtonStepPlan(predict,stage,finish,3,repetitions,
        i->begin push!(events,:audit);fail_audit && error("rejected by audit");end,
        0.,status,i->push!(events,:newton_audit)),repetitions,.001)
    return (;stepper,events,status)
end

@testset "resident Newton cursor pauses without replaying prediction or committing prefixes" begin
    for split in (false,true)
        fixture=cursor_test_stepper(;split)
        cursor=PS.ReusableNewtonCursor(fixture.stepper)
        expired=PS.advance_reusable_newton!(cursor;deadline=time()-1)
        @test expired.executed_stages==0 && expired.completed_steps==0
        @test isempty(fixture.events)
        @test_throws ArgumentError PS.advance_reusable_newton!(cursor;deadline=Inf)
        @test_throws ArgumentError PS.advance_reusable_newton!(cursor;deadline=time()+1,max_stages=0)
        @test_throws ArgumentError PS.advance_reusable_newton!(cursor;deadline=time()+1,stage_margin=-1)
        before=(cursor.phase,cursor.step,cursor.iteration)
        @test PS.advance_reusable_newton!(cursor;deadline=time()+10,stage_margin=20).executed_stages==0
        @test (cursor.phase,cursor.step,cursor.iteration)==before
        receipts=NamedTuple[]
        while cursor.phase!==:done
            before=length(fixture.events)
            receipt=PS.advance_reusable_newton!(cursor;deadline=time()+10,max_stages=1)
            push!(receipts,receipt)
            @test receipt.executed_stages==1
            @test length(fixture.events)>before
            @test count(==(:audit),fixture.events)==receipt.completed_steps
            # A second already-expired reservation must not launch anything.
            paused=PS.advance_reusable_newton!(cursor;deadline=time()-1)
            @test paused.executed_stages==0
            @test paused.next_phase==receipt.next_phase
        end
        @test last(receipts).complete
        @test count(==(:predict),fixture.events)==2
        @test count(==(:trial),fixture.events)==4
        @test count(==(:finish),fixture.events)==2
        @test count(==(:linear),fixture.events)==(split ? 4 : 0)
        @test count(==(:linear_audit),fixture.events)==(split ? 4 : 0)
        @test PS.advance_reusable_newton!(cursor;deadline=time()+10).executed_stages==0
    end
    invalid=cursor_test_stepper(;bad_receipt=true)
    cursor=PS.ReusableNewtonCursor(invalid.stepper)
    @test_throws ErrorException PS.advance_reusable_newton!(cursor;deadline=time()+10)
    @test cursor.poisoned
    @test !(:finish in invalid.events)
    @test_throws ErrorException PS.advance_reusable_newton!(cursor;deadline=time()+10)
    rejected=cursor_test_stepper(;fail_audit=true)
    cursor=PS.ReusableNewtonCursor(rejected.stepper)
    @test_throws ErrorException PS.advance_reusable_newton!(cursor;deadline=time()+10)
    @test cursor.poisoned
    @test cursor.step==1
    @test count(==(:predict),rejected.events)==1
    @test_throws ErrorException PS.advance_reusable_newton!(cursor;deadline=time()+10)
end
