using PrintSim, Test, LinearAlgebra
import PrintSim as PS

@testset "connectivity-local solves match dense principal blocks" begin
    b = ModelBuilder()
    points,tets = tet_box((0.008,0.006,0.002),(3,2,2))
    add_deformable!(b,points,tets; young = 1e5,density = 1000)
    m = PS.FEMModel(Array,b)
    s = PS.FEMSchwarz(Array,b)
    @test all(>(0),s.node_patch)
    @test sort([s.nodes[p,i] for p in 1:s.patches for i in 1:s.count[p]]) == collect(1:m.nodes)
    gradients = zeros(PS.Vec3f,7,10,m.elements)
    A = zeros(Float64,3m.nodes,3m.nodes)
    for i in 1:m.nodes,k in 1:3
        A[3i-3+k,3i-3+k] = m.mass[i]/1f-3^2
    end
    for e in 1:m.elements,a in 1:10
        g = m.gradient_hi[a,e]+m.gradient_lo[a,e]
        x,y,z = g
        gradients[:,a,e] = [PS.Vec3f(x,0,0),PS.Vec3f(0,y,0),PS.Vec3f(0,0,z),
            PS.Vec3f(y,x,0)/sqrt(2f0),PS.Vec3f(z,0,x)/sqrt(2f0),
            PS.Vec3f(0,z,y)/sqrt(2f0),g]
    end
    # Assemble the full dense rest tangent independently, then compare each
    # graph-selected principal block and its action against a Float64 solve.
    for e in 1:m.elements,a in 1:10,c in 1:10,k in 1:7
        rows,cols = 3m.tets[a,e].+(-2:0),3m.tets[c,e].+(-2:0)
        A[rows,cols] .+= Float64(m.stiffness[k,e])*Float64.(gradients[k,a,e])*Float64.(gradients[k,c,e])'
    end
    inverse = [PS.preconditioner_inverse(PS.Mat3f(A[3i-2:3i,3i-2:3i])) for i in 1:m.nodes]
    for i in 1:m.nodes
        PS.schwarz_assemble_node!(s,inverse,gradients,m,Int32(i))
    end
    r = [PS.Vec3f(sin(i),cos(i),sin(i/3)) for i in 1:m.nodes]
    z = zeros(PS.Vec3f,m.nodes)
    flat_r = reduce(vcat,r)
    for p in 1:s.patches
        nodes = s.nodes[p,1:s.count[p]]
        ids = [3i-3+k for i in nodes for k in 1:3]
        @test s.factor[p,1:length(ids),1:length(ids)] ≈ A[ids,ids] rtol = 2e-5
        PS.schwarz_factor_patch!(s,Int32(p))
        @test s.fallback[p] == 0
        PS.schwarz_apply_patch!(z,r,s,m,Int32(p))
        actual = reduce(vcat,z[nodes])
        @test actual ≈ A[ids,ids]\flat_r[ids] rtol = 3e-4
    end
    @test dot(flat_r,reduce(vcat,z)) > 0
end
