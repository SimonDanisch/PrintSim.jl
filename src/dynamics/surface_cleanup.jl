"Connected components and Euler characteristic of a closed, oriented triangle surface."
function closed_surface_topology(points, faces)
    parent = collect(eachindex(points))
    used = Set{Int}()
    edges = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    function root(v)
        while parent[v] != v
            parent[v] = parent[parent[v]]
            v = parent[v]
        end
        return v
    end
    for f in faces, j in 1:3
        a, b = convert(Int, f[j]), convert(Int, f[mod1(j + 1, 3)])
        push!(used, a, b)
        parent[root(b)] = root(a)
        edge = minmax(a, b)
        count, orientation = get(edges, edge, (0, 0))
        edges[edge] = count + 1, orientation + (a < b ? 1 : -1)
    end
    all(==((2, 0)), values(edges)) || error("surface is not a closed oriented edge manifold")
    return (; components = length(Set(root(v) for v in used)), euler = length(used) - length(edges) + length(faces))
end

function tetrahedral_boundary(points, tets)
    counts = Dict{NTuple{3, Int32}, Tuple{NTuple{3, Int32}, Int}}()
    for (a, b, c, d) in tets
        det(Matrix(hcat(points[b] - points[a], points[c] - points[a], points[d] - points[a]))) < 0 && ((c, d) = (d, c))
        for f in ((b, c, d), (a, d, c), (a, b, d), (a, c, b))
            key = triangle_node_key(f)
            old = get(counts, key, (f, 0))
            old[2] < 2 || error("nonmanifold volume")
            counts[key] = f, old[2] + 1
        end
    end
    return [f for (f, n) in values(counts) if n == 1]
end
