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
using GeometryBasics
using Makie
using Gmsh: gmsh
using Packing: GuillotinePacker
using ZipArchives: ZipWriter, zip_newfile, zip_writefile
import LibGEOS
import RayMakie, Hikari, Downloads
import Raycore, Mantle, Adapt
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize

export Item, Part, printed, parts, with_gmsh, occ, prism, box3, cylinder_x, cut!, fuse!
export Bed, PrintObject, export_3mf
export PrintedPart, read_beads, printed_parts, reprint, section
export Joint, BoughtSection, joint_report, key_report
export Rays, Traced, Pose, turn_x, shared, travel
export RayChecks, interference, collisions, sequence_check, axial_check
export AssemblyStep, Lead, Studio, assembly_guide, step_meshes

include("solids.jl")
include("print.jl")
include("printsim.jl")
include("joints.jl")
include("rays.jl")
include("guide.jl")
include("checks.jl")

end
