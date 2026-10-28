"""
Simulate, check and document 3D-printed assemblies before printing them.

An assembly is a list of `Item`s, each a solid built in Gmsh in its assembled place with its
material and print orientation. From that list PrintSim

- writes the print files: STEP and STL per part (`parts`), a Bambu Studio project with the
  parts packed onto plates (`export_3mf`);
- rebuilds each part as it prints from the slicer's G-code, swollen and shrunk
  (`printed_parts`, `reprint`), and checks where parts meet on those (`Joint`, `joint_report`);
- checks the assembly on the GPU with ray queries (`RayChecks`): what overlaps in place, what a
  move runs into, whether the parts go together in order (`sequence_check`), how far each
  slides along its shaft (`axial_check`);
- renders illustrated assembly instructions with RayMakie (`Studio`, `assembly_guide`).

    using PrintSim
    list = [Item("plate", :PLA, :steelblue, _ -> box3((0, 0, 0), (100, 60, 4)))]
    export_3mf(list, "assembly.3mf")
    c = RayChecks(list)
    interference(c)
"""
module PrintSim

using LinearAlgebra
import SHA
using GeometryBasics
using Makie
using Gmsh: gmsh
using Packing: GuillotinePacker
using ZipArchives: ZipWriter, zip_newfile, zip_writefile, ZipReader, zip_readentry
import LibGEOS
import RayMakie, Hikari, Downloads, Serialization
import Raycore, Mantle, Adapt
import GPUMeshing
using GPUMeshing: SparseGrid, DualPatches, dualmc, cellkey, patchcount, patchof, qefsolve, prefixsum, lastentry, runonce!
import KernelInterface as KI
using StaticArrays: MVector
import Atomix

# Device atomics below claim independent slots or combine reductions. Their
# payloads are consumed after the dispatch, rather than published to other
# lanes by the counter, so they use explicit monotonic ordering.

export Item, Part, printed, parts, with_gmsh, occ, prism, box3, cylinder_x, cut!, fuse!
export Bed, PrintObject, export_3mf
export PrintedPart, read_beads, printed_parts, reprint, section
export BeadSegments, read_segments, printed_surface, printed_surfaces
export PartPlacement, printed_placements, assembled!
export printed_mesh
export printed_project
export Joint, BoughtSection, joint_report, key_report
export Rays, Traced, Pose, turn_x, shared, travel
export RayChecks, interference, collisions, sequence_check, axial_check
export AssemblyStep, Lead, Studio, assembly_guide, step_meshes
export SheetMaterial, SheetGrid, SheetPiece, ModelBuilder, Model, State, SheetSolver, Stepper, add_sheet!, fix!, vertex, vertices
export Position, approximate, stack!, add_mesh!, box_mesh, resting_gap
export torus_mesh
export add_body!, AffineModel, AffineState, BodySolver, body_pose
export WorldSolver, WorldState
export add_joint!
export add_stepper!, command!, set_load!
export add_deformable!, tet_box, FEMModel, FEMState, FEMSolver
export tetrahedralize
export add_bond!, exclude_contacts!
export MotorSequence
export waitfor!
export Recording, record!, save_recording, load_recording, player, save_video
export state_checkpoint, save_state_checkpoint, restore_state_checkpoint!

include("solids.jl")
include("print.jl")
include("printsim.jl")
include("printed_mesh.jl")
include("printed_project.jl")
include("joints.jl")
include("launch.jl")
include("printed_surface.jl")
include("printed_placement.jl")
include("gpu_window.jl")
include("rays.jl")
include("guide.jl")
include("checks.jl")
include("dynamics/precise.jl")
include("dynamics/ccd.jl")
include("dynamics/accd.jl")
include("dynamics/model.jl")
include("dynamics/sheets.jl")
include("dynamics/vbd.jl")
include("dynamics/multigrid.jl")
include("dynamics/stack.jl")
include("dynamics/meshes.jl")
include("dynamics/contacts.jl")
include("dynamics/faces.jl")
include("dynamics/incidence.jl")
include("dynamics/affine.jl")
include("dynamics/joints.jl")
include("dynamics/coupled.jl")
include("dynamics/motors.jl")
include("dynamics/body_energy.jl")
include("dynamics/fem_gradients.jl")
include("dynamics/tetrahedra.jl")
include("dynamics/assembly_grip.jl")
include("dynamics/surface_embedding.jl")
include("dynamics/quadratic_proxy.jl")
include("dynamics/tet_mesh.jl")
include("dynamics/surface_cleanup.jl")
include("dynamics/simplify.jl")
include("dynamics/fem_coarse.jl")
include("dynamics/fem_schwarz.jl")
include("dynamics/fem_contacts.jl")
include("dynamics/mapped_fem.jl")
include("dynamics/volume_embedding.jl")
include("dynamics/proxy_guard.jl")
include("dynamics/swept_ccd.jl")
include("dynamics/ccd_projection.jl")
include("dynamics/prefix_scan.jl")
include("dynamics/ccd_queue.jl")
include("dynamics/ccd_parallel.jl")
include("dynamics/ccd_stream.jl")
include("dynamics/bonds.jl")
include("dynamics/world_energy.jl")
include("dynamics/newton_audit.jl")
include("dynamics/control.jl")
include("dynamics/recording.jl")
include("dynamics/recording_surface.jl")
include("dynamics/checkpoint.jl")

end
