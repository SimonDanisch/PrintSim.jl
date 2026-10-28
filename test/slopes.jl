# What the tests lay their sheets on: a board, meshed.

"""
    slope!(builder, θdeg; static, kinetic) -> (up, across, normal)

A 0.4 m square board tilted by `θdeg` about y, its top face through the origin, meshed in
Float64 and added to `builder`; returns the face's frame, exactly orthonormal in Float64, to lay
sheets along.
"""
function slope!(b, θdeg; static = 0.5, kinetic = 0.4, half = 0.2, depth = 0.01)
    θ = deg2rad(θdeg)
    up, across, normal = Vec3(cos(θ), 0.0, sin(θ)), Vec3(0.0, 1.0, 0.0), Vec3(-sin(θ), 0.0, cos(θ))
    add_mesh!(b, box_mesh(-depth / 2 * normal, (up, across, normal), (half, half, depth / 2)); static, kinetic)
    return up, across, normal
end
