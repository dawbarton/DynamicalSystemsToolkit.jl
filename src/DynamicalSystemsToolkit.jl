module DynamicalSystemsToolkit

using ModelingToolkit
using RuntimeGeneratedFunctions
using DocStringExtensions
using LinearAlgebra

const MTK = ModelingToolkit

RuntimeGeneratedFunctions.init(@__MODULE__)

# Exports

export CollocationSystem

# Abstract types

abstract type AbstractDynamicalSystem <: MTK.AbstractSystem end

# Subsystems

include("collocation.jl")

# Utilities

function bind_parameters!(basesys::AbstractDynamicalSystem, sys1::AbstractDynamicalSystem, sys2::AbstractDynamicalSystem, ps)
end

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
