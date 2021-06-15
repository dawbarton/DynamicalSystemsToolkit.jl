function periodic_orbit(
    T::Type, sys::ODESystem, n_mesh::Integer, n_coll::Integer; name=gensym(:PeriodicOrbit)
)
    collsys = collocation(T, sys, n_mesh, n_coll; name)
    return push!(collsys.equations, collsys.t₀ ~ 0)
end

function periodic_orbit(sys::ODESystem, args...; kwargs...)
    return periodic_orbit(Float64, sys, args...; kwargs...)
end
