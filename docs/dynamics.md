# PrintSim dynamics: printed mechanisms and paper in 3D, on the GPU

Target scope: the printed parts as they come off the printer (rebuilt from the slicer's G-code),
their shafts, bearings, keys, clips, gears, pawls and printed springs, the bought parts, the
motors that drive them, the firmware, and thin flexible sheets (paper) that the mechanism
handles. Everything is simulated together in 3D on the GPU through Mantle, so it runs on any
GPU Mantle supports, recorded, and played back in a RayMakie dashboard.

## The whole feature set, planned at once

Built in steps, but every step lands in this design, so nothing is bolted on later.

**Everything is a triangle mesh.** A contact only ever sees triangles: as-printed meshes of
printed parts (their print errors included, so a bad print shows up in the simulation), the
CAD of bought parts, and simple geometry from GeometryBasics (a `Torus` for an O-ring, a
`Cylinder` for a steel shaft) meshed the same way. No analytic shapes, no feeder-specific
obstacles. Curved bought parts use fine meshes, with chord error below the contact roughness:
for a circular cross-section of radius R, R(1 − cos(π/N)) ≤ σ for N facets. At R = 6 mm and
σ = 1 µm, 256 facets give 0.45 µm error; 32 give 28.89 µm. The settling study below checks
stability separately from this geometric accuracy. No interpolated contact surface is needed
for that test. Printed facets and bead ridges are real and stay.

**Objects whose surfaces are linear in their degrees of freedom.** Three kinds, one interface:
each gives surface vertices and triangles whose positions are a linear function of its DOFs,
so contacts, their Jacobians and the solver treat all alike.

- *Sheets*: shells on structured grids (below); a surface vertex is a DOF.
- *Stiff parts* (gears, shafts, rollers, frames): affine bodies (Lan, Kaufman, Li, Jiang, Yang,
  *Affine Body Dynamics*, ACM TOG 41(4), 2022): 12 DOFs (A, t), x = A X + t, a stiff energy
  keeping A orthogonal. A contact's block maps to the body through a 3×12 matrix, and the
  body's own block is a dense 12×12.
- *Compliant parts* (the pad's PETG leaf, pawls, printed springs, and later any part whose
  stress is checked): finite elements on tetrahedra meshed from the as-printed geometry, with
  the printed material's elasticity. Stresses come out per element, which a later strength or
  failure check needs. Each part picks affine or FE in the builder; both share contacts and
  solver.
- *Joints* only where no geometry is modeled (a motor shaft in its housing): stiff springs
  between DOFs. Bearings with play, keys, clips, gear meshes, pawls on ratchets are contacts
  between the parts' meshes.
- *Motors*: steppers driving their shaft through a joint, with the measured torque and pull-out
  behavior, so steps get lost under too much load by themselves.

**Contacts**, one pipeline for every pair of objects: a BVH over all triangles (Raycore, with
Mantle's hardware ray tracing where the device has it), refit every step; vertex–triangle and
edge–edge pairs, distances taken in double-float from double-float DOFs; penalty with an
augmented-Lagrangian update (as in AVBD) so stiff contacts hold without huge stiffness;
Cundall-Strack friction with contact memory. Sheets find each other through the same query
(a stack, sheets landing in the box, a fold), not through fixed neighbors.

**Solver**: one implicit step for all DOFs. A Newton step on the incremental potential of
everything, solved by PCG with a preconditioner built from blocks: multigrid for sheets (with
the couplings between stacked sheets on every level, below), a dense 12×12 solve per affine
body, multigrid or block-Jacobi for tetrahedral bodies, the couplings between objects carried
as measured. Alternating solvers per object kind (AVBD for bodies) only if measured faster.

**Control, recording, dashboard**: the firmware as a state machine stepped per frame; motor
commands as device arrays so recorded Mantle graphs run without the host. Every frame keeps
sheet vertices, body poses and tetrahedral positions, contact and motor quantities; the
dashboard (RayMakie, rasterized) moves the parts' own meshes, colors by force, slip or stress,
and draws the quantities over time.

**The feeder** comes entirely from `FeederDesign` and `Build`: every item of the assembly as
its mesh (as printed, at a chosen print corner), its kind by its role, the parts pressed or
glued together as one body, motors at their pinions, the firmware's sequence.

**Order of building**, each step validated before the next, each watchable in the dashboard:

1. Contacts on meshes: the sheet tests below again with meshed planes and cylinders; the facet
   question for curved parts.
2. Affine bodies and body–body contacts: a block sliding on a slope, a gear pair with backlash
   against the print simulation's joints.
3. One system: a roller driving a sheet stack, the pick arm resting on a stack.
4. Tetrahedral bodies: the pad's leaf force (0.9 N) against beam theory, pawl drag against
   `pawl_check`.
5. Motors and firmware: a stepper loses steps at its pull-out torque.
6. The feeder: against the 2D model's validation (`validate.jl`) and its envelope cases.

## Reference

The architecture and the solver follow [Newton](https://github.com/newton-physics/newton)
(Apache-2.0; NVIDIA, Google DeepMind, Disney Research), the current reference for GPU
simulation of cloth coupled with rigid bodies:

- **Data flow** (`newton/_src/sim`): a `ModelBuilder` on the host collects particles, sheets,
  bodies, shapes, joints and actuators and finalizes them into a `Model` on the device: static
  description, never written during a run. A `State` holds what changes (particle positions
  and velocities, body poses and velocities); a `Control` holds the inputs (motor commands);
  `Contacts` hold what the collision pipeline found. A solver advances
  `step!(solver, state_in, state_out, control, contacts, dt)`.
- **Particles: Vertex Block Descent** (Chen, Liu, Yang, Yuksel, *Vertex Block Descent*, ACM
  TOG 43(4), 2024). Implicit Euler as minimization of the incremental potential; each vertex
  in turn takes a Newton step on its own 3x3 block, vertices of one graph color in parallel.
  Membrane: stable Neo-Hookean (Smith et al. 2018) on triangles; bending: dihedral angle per
  interior edge with rest angle; contact: offset penalty with IPC-style lagged friction.
- **Rigid bodies: Augmented VBD** (Giles, Diaz, Yuksel, *Augmented Vertex Block Descent*, ACM
  TOG 44(4), 2025). Each body is a 6-DOF block; joints and contacts are constraints with
  augmented-Lagrangian multipliers, so stiff joints and stacked contacts hold without tiny
  steps. Body-particle contacts are shared by both sides, as in Newton's `SolverVBD`.

Where the paper path needs more than Newton offers, it is said below and measured.

## What differs from Newton, and why

### Coupled recording layout

`Stepper(world, state, dt, steps; reusable_newton=true)` records prediction,
one Newton stage, and final velocity/friction updates separately. The host
replays the same resident Newton plan; device buffers retain CG state,
convergence flags and the executed-Newton index. This avoids retaining one
copy of every recorded command buffer for each requested Newton iteration.
Motor control advances once during prediction, not once per Newton replay.
After a synchronized adaptive Newton stage, the host reads only the GPU's
two-word execution/convergence receipt to avoid submitting further inactive
stages. The convergence calculation and all physical solves remain on device.
The first implementation synchronizes between stages; it is not an
asynchronous-overlap or full-feeder throughput result.

`split_newton=true` additionally separates the resident linear solve/contact
bound from CCD/line search. `on_linear(iteration)` runs after synchronization,
before trial submission; the deadline is checked again after that callback.
`on_prediction()` reports completed prediction rather than merely graph
construction. Callback rejection or deadline expiry does not finalize/commit
the timestep. Twenty-four Radeon comparisons cover split/unsplit parity;
the actual feeder-driver fixture also passes 21 hardware cold-resume checks.
The feeder's `FEEDER_SPLIT_NEWTON` and `FEEDER_TRACE_LINEAR` flags select this
diagnostic mode. Linear tracing persists scalar solve evidence and, if its
work margin permits, the first contact-bounded collision input of each step.
It is not intended for whole-run snapshot storage.

`compact_gradients=true` reconstructs the existing seven invariant gradients
from three cached Float32 deformation columns and existing rest shape
gradients. The first node retains the dense kernel's ordered negative sum.
`matrix_free_blocks=true` additionally reconstructs the diagonal quadrature
blocks during RHS/preconditioner assembly, avoiding ten stored dense 3×3
matrices per quadrature point. CG still uses the element-wise product. This
avoids 720 bytes per quadrature point in FP64 (360 in Float32); it is an
allocation reduction, not a measured full-feeder peak-memory or speed claim.
Seventy-five RX 7900 XTX checks cover FP64 operators/preconditioners, reduced
proxy mappings, Float32 block equivalence and complete coupled replay.
No material law, quadrature, displacement precision, tangent, or AMG hierarchy
changes. For 1,434,275 T10 cells (four quadrature points each), gradient storage
falls from 4,819,164,000 to 206,535,600 bytes; shared rest gradients are not
new storage. Fifty comparisons on both software Vulkan and Radeon cover forces,
tangent products, Schwarz factors and coupled Newton replay. The dense
representation remains the numerical parity reference; full-scene qualification
is still separate.
The feeder exposes `FEEDER_COMPACT_FEM_GRADIENTS` (default false).

The feeder now defaults `FEEDER_VALIDATE_T10=true`: the existing licensed native
HOCGV adapter certifies the initial whole-cell Jacobians and the complete linear
nodal path between every consecutive accepted timestep. Quadrature-only and
endpoint-only tests are insufficient. Checker uncertainty, timeout, conversion
error or inversion stops the audit; the timestep is not added to the accepted
recording/checkpoint chain. This is a slow independent reference path, not a
claim of GPU validity-filter performance. Protocol 2 transmits both stored
coordinate limbs independently. Native outward interval addition encloses their
exact sum before evaluating the pinned upstream's generated Jacobian expressions
and its unchanged subdivision algorithm. Minimal upstream coefficient-entry
overloads avoid narrowing those inputs to Float64. The old scalar protocol
remains readable for historical replay; new writes and reports use protocol 2.
Forty-four native checks include previously unrepresentable coordinates, folds,
between-endpoint inversion and certificate-chain integrity. The independent
Geogram scalar adapter still refuses unrepresentable inputs rather than rounding.

Per-step reports include input/checker hashes and endpoint-chain digests. Failed
inputs are retained; successful inputs are deleted after the certificate is
committed (otherwise this assembly would retain tens of TB). `retain_input=true`
keeps selected inputs for independent replay. Cold continuations certify their
restored start state, and export re-reads the complete consecutive certificate
chain along with all linear/nonlinear solve reports. Skipping T10 validation in
a diagnostic never qualifies that recording for export. Neither this Jacobian
certificate nor successful solves establish physical surface-envelope validation.

This layout is opt-in pending full-scene qualification. A
sheet/T10 fixture passes 42 software-Vulkan comparisons against the original
unrolled layout, including two replays with fixed and adaptive Newton counts.
That fixture deliberately isolates recording layout from moving CCD; it does
not qualify the latter or establish a completed feeder timestep.

Both layouts reject plans exceeding `recording_limit=4096` command-buffer
pieces before opening command buffers. The limit is an allocation guard, not
a proof that smaller recordings fit a device's memory. Likewise the pidfd
watchdog cannot terminate a worker stuck in an uninterruptible driver wait.

The feeder entry point exposes `FEEDER_REUSABLE_NEWTON` (default `false`) and
`FEEDER_RECORDING_LIMIT`. Keep normal collision, solve and geometric audits
enabled when selecting either layout.

Each point was measured on paper (80 g/m², E·t ≈ 4·10⁵ N/m, D ≈ 3.7·10⁻⁴ N·m).

- **The solver.** Vertex Block Descent needs steps of 5 µs for paper: a vertex's inertia m/dt²
  is 0.5 N/m at dt = 1 ms against 10⁶ N/m of membrane and 10³ N/m of bending, so each sweep moves
  a vertex a fraction of what the elastic forces ask and smooth bending travels across the sheet
  over thousands of sweeps. A 50 mm cantilever then hangs at 25 to 50 mm instead of 1.8 mm.
  Sweeping whole grid lines as blocks does no better (the hinges hold a line to its neighbors).
  Instead each step is a Newton step on the same incremental potential, solved by conjugate
  gradients preconditioned with a geometric multigrid V-cycle on the sheets' grids, Galerkin
  coarse levels stored as 5×5 stencils of 3×3 blocks (`multigrid.jl`): 13 iterations where
  block-Jacobi needs 2160, a rediscretized hierarchy 635 (65×33 vertices). Tamstorf, Jones and
  McCormick (2015) solve the same stiff cloth systems with algebraic multigrid. Two details make
  it hold for paper: products take the difference form Sᵢxᵢ + Σⱼ Aᵢⱼ(xⱼ − xᵢ), so moving a sheet
  rigidly, which the membrane blocks (10⁵ times a vertex's inertia) do not resist, is not lost in
  Float32 cancellation; and the coarsest level is solved by a dense Cholesky per stack, because
  Jacobi removes a millionth of the rigid in-plane error per sweep (a falling sheet tipped 35°
  fell in its plane at 4.48 instead of 4.61 m/s² with 12 iterations; with the direct coarse solve
  exactly, on every grid up to a whole A4 sheet).
- **The grid** splits every cell along the same diagonal: with alternating diagonals the
  vertices of four and of eight neighbors get opposite forces under even bending and the sheet
  buckles into a zigzag.
- **In-plane elasticity** is springs along edges and diagonals (paper strains by 10⁻⁴; its
  constitutive law in the plane does not matter), with geometric stiffness only under tension:
  Newton's projected Neo-Hookean block keeps the membrane's stiffness across the sheet too and
  freezes paper in its plane.
- **Positions and velocities are double-float** (`precise.jl`): Float32 resolves 0.05 m to 4 nm,
  a paper vertex rests 0.04 nm deep, and the contact forces came out as noise of 20 times a
  vertex's weight. What is taken from them is taken in double-float too: spring stretches
  (Float32 rounds a 2 mm edge to 0.1 nm, which the membrane turned into forces of 8 times a
  vertex's weight and a stuck sheet into sliding), the step a velocity carries a vertex (rounded
  alike for neighbors, it shifted two sheets sliding together against each other by as much as
  they press into one another), and distances along a Float32 normal, which is unit only to
  10⁻⁷ (6 pm over a sheet's thickness, different for each triangle: the normal force between
  sliding sheets jumped by 10 % from step to step). TwoProd uses Dekker's split, not `fma`, which
  Vulkan lets a driver evaluate as a product and a sum (lavapipe does); Lava marks Julia's
  arithmetic `NoContraction`, without which Mesa folds the error terms away.
- **Contact** measures the gap to a triangle mesh grown by half the sheet's thickness, in
  double-float. Both sheet vertices against mesh triangles and convex mesh features against
  sheet faces contribute, so narrow rails between grid lines still support the sheet. Normal
  pressure is k σ log(1 + exp(−gap/σ)), with σ = 1 µm and bulk stiffness k = 2·10¹⁰ Pa/m by
  default. It grows exponentially as rough surfaces approach and linearly once pressed together.
  At light loads its stiffness is pressure/σ, so increasing the bulk stiffness does not make a
  resting sheet chatter between facets in the roller test below. Every step starts each vertex
  where its velocity alone takes it, not from the inertial prediction: there gravity puts a
  resting vertex 10 µm into what it rests on, the
  right-hand side holds 0.4 N of contact force against µN of physics, and the rounding came out
  as normal forces several times the weight and sliding faster than gravity allows. Contact
  search includes predicted motion; it does not move vertices onto the mesh.
- **Friction** is a tangential spring capped by Coulomb's law (Cundall and Strack, as in the
  2D model) with separate static and kinetic coefficients. Its stiffness is half the current
  normal stiffness. The force follows the spring plus the current step's tangential motion,
  capped at μs λ while sticking and μk λ while sliding; shortening the spring allows sticking
  again. Using only the previous slip direction overshot slow sliding and made stacked sheets
  accelerate on a slope below their friction angle. IPC's smooth friction, which Newton's VBD
  kernels use, needs Newton iterated to convergence: from one
  linearization it acts as a viscosity, a sheet on a slope below its friction angle creeps at
  centimeters per second, and one starting to slide is held.
- **Sheets touch their neighbors in a stack.** Contacts between sheets are vertex-triangle
  between sheets adjacent in the stack, taken once from the upper sheet, with the same penalty
  and friction as on meshes. The bulk contact stiffness can be 10⁵ times a vertex's inertia,
  so contacts are part of the step's matrix on every multigrid level (a second 5×5 stencil per
  vertex into the sheet below, the transposed blocks gathered from the sheet above), each grid position's column
  through the stack is smoothed as one block-tridiagonal system, and the coarsest level is
  solved exactly over the whole stack. Treated as walls by the preconditioner, a sliding pair
  needed 88 iterations and 8 sheets more than 500; now 13 and 18 (condition 4.5 and 4.9). Every
  row is gathered by its own vertex, no atomics, so a run repeats itself.

## Units and precision

SI units inside the solver (m, kg, s). Positions and velocities use double-float
(pairs of Float32). Float32 remains the default solver arithmetic. On devices
with native FP64, `elastic_krylov_precision=:float64` selects FP64 elastic
vectors, tangent gradients, local block factors and CG reductions. The idler
assembly harness also uploads its AMG matrices, transfers and coarse factors
in FP64. Rest-gradient high and low limbs are both retained; coarse restriction
and CG use two-pass reductions without requiring FP64 atomic addition. This
does not change the rest geometry, material constants or physical tolerances.
Focused RX 7900 XTX checks compare products and preconditioner corrections to
host references. They do not establish convergence of the printed feeder.

## Verification

Each layer is checked before the next is built on it:

Every test runs on the machine's GPU (Mantle's default device, here a Radeon 8060S).

1. Sheet (`test/sheets.jl`): a 50 mm cantilever hangs 1.93 mm on a 2.5 mm grid, 1.87 on
   1.25 mm, between plate theory (1.67, edges held) and beam theory (1.84, edges free); the
   device solver follows an exact sparse solve of every step (double-float positions too) to
   20 µm.
2. Sheet on meshes (`test/contacts.jl`, `test/faces.jl`): a sheet on a meshed plane carries its
   weight to 0.2 %. On a 20° slope with μs = 0.5 its friction springs give way by
   2σ tan θ ≈ 0.73 µm, then it rests (less than 0.1 nm over the final ten steps). On 35° with
   μk = 0.4 its acceleration matches g(sin θ − μk cos θ) to 0.2 %. Two 1 mm rails between
   grid lines support its faces, carry its weight to 1 %, and keep the sag below 0.1 mm.
3. Sheet on sheet (`test/stack.jl`): two sheets on a plane: the plane carries 2.000 sheet
   weights, the lower sheet the upper one's 1.000 (within 0.5 % after settling). On 20° the
   contacts give way, then both sheets rest (less than 0.1 nm over the final ten steps).
   On 35° both accelerations match the gravity and friction balance to 0.5 %. Six sheets on
   20° carry the sheets above them to 0.5 % and move less than 2 nm in the final ten steps.
   Still to do: a stack of curled sheets settles without passing through itself; pulling the
   top sheet needs μ·N.
4. Bodies: gear ratio and backlash against the print simulation's joints; pawl drag against
   `pawl_check`; a stepper loses steps at its pull-out torque.
5. Coupled: the nip pull test of the 2D model's validation (`validate.jl`).
6. The feeder: motor forces, timing and skew against the 2D model and the skew model.

### Faceted roller study

`test/facets.jl` repeats the 50 mm sheet settling on a stationary 12 mm roller for 600 steps
of 1 ms, with 17×17 and 33×33 sheet vertices and bulk stiffness 2·10⁹ and 2·10¹⁰ Pa/m.
All 16 combinations on the Radeon remain stable. The table reports the largest absolute
deviation of the summed contact magnitudes from the sheet weight over steps 300–600, across
both grids and stiffnesses, and the smallest sampled surface clearance at the final step.

| Roller facets | Circle chord error (µm) | Largest load error (%) | Smallest clearance (µm) |
| --- | ---: | ---: | ---: |
| 32 | 28.892 | 0.00018 | 4.036 |
| 64 | 7.227 | 0.00020 | 4.036 |
| 128 | 1.807 | 0.02720 | 4.036 |
| 256 | 0.452 | 0.04458 | 4.123 |

These results validate stationary settling and feature contacts, including edges between
sheet vertices. They do not yet validate a rotating roller or a loaded nip. The geometric
criterion above still calls for finer meshes on bought round parts even when a coarse mesh
settles stably. `test/rollers.jl` retains the coarse and fine endpoint cases as regressions.
The regression bound is 0.5 %, as for the stack load tests.

Run the study from the workspace environment:

```sh
julia --project=. dev/PrintSim/test/facets.jl
```
