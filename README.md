# PrintSim

Simulate, check and document 3D-printed assemblies before printing them.

An assembly is a list of `Item`s: a solid built in Gmsh/OpenCASCADE in its assembled place,
with material, color and print orientation. From that list PrintSim

- writes STEP and STL per part (`parts`) and a Bambu Studio project with the parts packed
  onto plates (`export_3mf`);
- rebuilds every part as it prints from the slicer's G-code, swollen and shrunk
  (`printed_parts`, `reprint`), and checks shafts, bores, keys and clips on those
  (`Joint`, `joint_report`, `key_report`);
- checks the assembly on the GPU with Raycore ray queries, on any backend Mantle supports
  (`RayChecks`): what overlaps in place (`interference`), what a move runs into
  (`collisions`), whether the parts go together in order (`sequence_check`), how far each
  slides along its shaft (`axial_check`);
- renders illustrated assembly instructions with RayMakie (`Studio`, `assembly_guide`).

```julia
using PrintSim
list = [Item("plate", :PLA, :steelblue, _ -> box3((0, 0, 0), (100, 60, 4)))]
export_3mf(list, "assembly.3mf")
c = RayChecks(list)
interference(c)
```

The paper feeder in [Paperhawk](https://github.com/SimonDanisch/Paperhawk.jl) (`hardware/feeder`)
is the first assembly built with it.
