# PrintSim dynamics: a port of the GPU state of the art to Mantle

Goal: simulate the as-printed paper feeder, meaning printed rigid parts, compliant printed
parts, paper sheets, bearings, steppers, glued and pressed joints, contact and friction. The
method is a faithful port of libuipc (Apache-2.0, `dev/libuipc`), whose CUDA backend unifies
the GPU IPC line of work:

- IPC (Li et al. 2020)
- C-IPC (Li et al. 2021)
- ABD (Lan et al. 2022)
- GIPC (Huang et al. 2024)
- StiffGIPC (Huang et al. 2025, `dev/Stiff-GIPC`)

It runs on Mantle and on the GPU only. The host builds the scene and moves nothing in the
solver loop.

## 1. Degrees of freedom

| Part | Model | libuipc source | PrintSim today | Action |
|---|---|---|---|---|
| Rigid printed parts, shafts, gears | Affine body, 12 DOF, orthogonality energy (ABD) | `affine_body/abd_energy.cu`, `affine_body_constitution.cu` | `affine.jl`, ABD-like with 6 orthogonality rows | Verify the energy against `abd_energy.cu`; keep if equal |
| Compliant printed parts (clips, springs, cup gears, pad spring) | FEM on linear tetrahedra, Stable Neo-Hookean | `finite_element/constitutions/stable_neo_hookean*`, `fem_3d_constitution.cu` | `tetrahedra.jl`: T10, StVK, Gauss–Newton | Port Stable NH. Element order is open question 2 |
| Paper | Codimensional shell: Neo-Hookean or Baraff–Witkin membrane, discrete-shell bending, thickness | `finite_element/codim_2d_constitution.cu`, `constitutions/neo_hookean_shell*`, `discrete_shell_bending*` | `sheets.jl`, `model.jl`: springs and hinges with their own solver paths | Replace |
| Fixed scenery | Static surface | contact only | static meshes | Keep |

Vertex positions are x = J q (affine bodies) or the nodes themselves (FEM, shells). Every
energy below is written over vertex positions, and each body type supplies only its Jacobian.

## 2. Energies (incremental potential)

| Term | Method | libuipc source | PrintSim today | Action |
|---|---|---|---|---|
| Inertia, gravity | Implicit Euler (BDF1) incremental potential | `time_integrator/`, `*_time_integrator.cu` | Implicit Euler | Keep, in Float64 |
| Contact | IPC log barrier on point–triangle, point–edge, point–point and edge–edge (mollified) pairs, codim thickness offsets, adaptive κ | `contact_system/simplex_normal_contact.cu`, `contact_models/ipc_simplex_normal_contact.cu`, `codim_ipc_*`, `adaptive_contact_parameter_reporter.cu` | Barrier on point–triangle only, spread over 5 separate paths | Replace with one simplex contact system |
| Friction | IPC lagged smooth friction (μ, ε_v), mollified EE pairs skipped | `contact_models/ipc_simplex_frictional_contact.cu` | Custom spring plus Coulomb | Replace |
| Bearings | Affine-body revolute joint | `constitutions/affine_body_revolute_joint` (docs: specification/constitutions) | `joints.jl` | Replace |
| Steppers | Driving revolute joint | `affine_body_driving_revolute_joint` | `motors.jl` | Replace |
| Press fits, glue | Fixed joint; soft vertex–triangle stitch | `affine_body_fixed_joint`, `soft_vertex_triangle_stitch` | `bonds.jl`; pinion modeled as a separate body | Replace |
| Grips and assembly tools | Soft position / transform constraint (animator) | `animator/`, `soft_position_constraint` | `assembly_grip.jl`: explicit force | Replace |

## 3. Solver

| Step | Method | libuipc source / default | PrintSim today | Action |
|---|---|---|---|---|
| Newton | Projected Newton with Gauss–Newton barrier Hessian (GIPC). Stop on velocity tolerance (`newton/velocity_tol` 0.05 m/s) plus semi-implicit stop (K_min 6, β 1e-3) | `newton_tolerance/`, `scene_default_config.cpp` | Newton, 10 nm correction tolerance | Port |
| Linear solve | Fused PCG on the global system (`tol_rate` 1e-3); ABD 12×12 block preconditioner; FEM diagonal block or MAS | `linear_system/linear_fused_pcg.cu`, `affine_body/abd_diag_preconditioner.cu`, `finite_element/fem_mas_preconditioner.cu` | PCG with custom Schwarz and coarse spaces | Port |
| CCD | Trajectory filter on the BVH (additive CCD), step scaled for safety | `collision_detection/filters/*simplex_trajectory_filter*` | Tight-Inclusion plus additive CCD (`accd.jl`) | Port the filter; keep `accd.jl` if it matches |
| Line search | Backtracking on the total energy, at most 8 halvings | `line_search/line_searcher.cu` | Backtracking on Float32 energy sums | Port in Float64 |
| Precision | Float64 throughout | `uipc/common/type_define.h` | Float32 with double-float positions | Float64 on devices with FP64 (Radeon); double-float where FP64 is missing (Metal) |
| Broad phase | Stackless LBVH | `collision_detection/info_stackless_bvh.h` | Own BVH | Port, or keep if equivalent |

## 4. Kept as is (not simulation methods)

- Print reconstruction, simplification, fTetWild meshing (geometry pipeline).
- Recording and RayMakie rendering. The player's camera override needs fixing.

## 5. Build order: each step working on the GPU against a known answer before the next

1. Affine bodies, static meshes, simplex contact (all four pair types), CCD, Newton, PCG.
   Block resting on a plate (load = weight); bar resting crosswise on a bar (edge–edge);
   a stack of blocks; two gears meshing.
2. Friction: block on an incline sticks below tan θ = μs and slides at g(sin θ − μk cos θ).
3. Joints and motors: pendulum period on a revolute joint; motor torque against a known
   inertia; gear train ratio; pinion fixed to its shaft.
4. FEM: PETG cantilever deflection against beam theory (F = 3EIδ/L³); snap fit over a pin;
   measured PCG iterations for PETG with diagonal and MAS preconditioners.
5. Paper shell: cantilevered sheet deflection against plate bending (D = Et³/12(1 − ν²));
   sheet resting at its thickness offset; roller nip pulling a sheet.
6. The 13 preload insertions (8 clips, 2 idler springs, 2 pawls, pinion as a fixed joint),
   then the feeder.

## 6. Open questions (decided by measurement during the build)

1. **Stiff PETG preconditioning.** libuipc's default FEM preconditioner is diagonal. On the
   idler spring's cage at dt = 1 ms that needs about 5,000 PCG iterations (κ ≈ 1e8). Two-level
   Schwarz needs about 200. MAS still has to be measured on the GPU after the port, and a
   larger dt makes the conditioning worse.
2. **Element order.** libuipc's FEM is linear tetrahedra. Thin printed leaves lock in bending
   with linear tets, so they need several elements across their thickness. Measured against
   beam theory in step 4.
3. **Contact surface.** In libuipc contact acts on the FE boundary. PrintSim currently embeds
   a finer surface in a coarse cage. The SOTA option is to mesh each cage boundary within the
   20–30 µm geometry tolerance and drop the embedding.
4. **Run length.** Two firmware cycles are 84 s. libuipc examples use dt = 10 ms. The step
   size is chosen from steps 1–5.
