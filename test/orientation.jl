using PrintSim,Test
import PrintSim as PS

@testset "zero StVK strain cannot disguise an inverted solid" begin
    b=ModelBuilder()
    points,tets=tet_box((0.003,0.003,0.001),(2,2,2))
    add_deformable!(b,points,tets)
    model=PS.FEMModel(Array,b)
    state=PS.FEMState(Array,b)
    @test all(isfinite(PS.fem_potential(state,model,e,1f0)) for e in 1:model.elements)
    for i in eachindex(state.x)
        p=b.deformables[1].initial[i]
        state.x[i]=Position(PS.Vec3d(-p[1],p[2],p[3]))
    end
    @test all(isinf(PS.fem_potential(state,model,e,1f0)) for e in 1:model.elements)
    # A proper 180-degree rotation has the same zero strain but positive
    # orientation, so the rejection must distinguish it from reflection.
    for i in eachindex(state.x)
        p=b.deformables[1].initial[i]
        state.x[i]=Position(PS.Vec3d(-p[1],-p[2],p[3]))
    end
    @test all(isfinite(PS.fem_potential(state,model,e,1f0)) for e in 1:model.elements)

    rigid=ModelBuilder()
    axes=(PS.Vec3d(1,0,0),PS.Vec3d(0,1,0),PS.Vec3d(0,0,1))
    add_body!(TEST_DEVICE, rigid,box_mesh(PS.Vec3d(0),axes,(0.003,0.002,0.001)))
    affine=PS.AffineModel(Array,rigid)
    pose=PS.AffineState(Array,rigid)
    @test isfinite(PS.affine_energy(pose.q,affine,Int32(1),1f0))
    for i in 2:4
        p=pose.q[i]
        pose.q[i]=Position(PS.Vec3f(-p.hi[1],p.hi[2],p.hi[3]),
            PS.Vec3f(-p.lo[1],p.lo[2],p.lo[3]))
    end
    @test isinf(PS.affine_energy(pose.q,affine,Int32(1),1f0))
end
