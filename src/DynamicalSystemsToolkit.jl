module DynamicalSystemsToolkit

using ModelingToolkit
using RuntimeGeneratedFunctions
using DocStringExtensions
using LinearAlgebra

const MTK = ModelingToolkit

RuntimeGeneratedFunctions.init(@__MODULE__)

# Exports - CollocationSystem

export CollocationSystem, add_boundarycondition!, generate_initialconditions

# Exports - DynamicalSystem

export DynamicalSystem

# Abstract types

abstract type AbstractDynamicalSystem <: MTK.AbstractSystem end

# Subsystems

include("collocation.jl")

# Utilities

function _strip_toplevel(name::String)
    idx = findfirst(==('₊'), name)
    return idx === nothing ? name[1:0] : name[nextind(name, idx):end]
end
_strip_toplevel(name::Symbol) = Symbol(_strip_toplevel(String(name)))


# General DynamicalSystem

struct DynamicalSystem <: AbstractDynamicalSystem
    eqs::Vector{Equation}
    states::Vector
    ps::Vector
    name::Symbol
    systems::Vector
    defaults::Dict
end

function DynamicalSystem(
    eqs, states, ps; name=gensym(:DynamicalSystem), defaults=Dict(), systems=[]
)
    sysnames = nameof.(systems)
    if length(unique(sysnames)) != length(sysnames)
        throw(ArgumentError("System names must be unique."))
    end
    defaults = MTK.todict(defaults)
    defaults = Dict(value(k) => value(v) for (k, v) in pairs(defaults))
    return DynamicalSystem(eqs, value.(states), value.(ps), name, systems, defaults)
end

end
