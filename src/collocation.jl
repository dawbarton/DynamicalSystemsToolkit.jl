cheb_mesh(T::Type, N::Integer) = (-cos.((T(π) * (0:N)) ./ N) .+ 1) ./ 2

function cheb(T::Type, N::Integer)
    if N == 0
        return (ones(T, 1), zeros(T, 1, 1))
    else
        x = cheb_mesh(T, N)
        c = [2; ones(T, N - 1, 1); 2] .* (-1) .^ (0:N)
        dx = x .- x'
        D = (c * (1 ./ c)') ./ (dx + I)
        D[diagind(D)] .-= vec(sum(D; dims=2))
        return (x, D)
    end
end

struct CollocationSystem{T} <: AbstractDynamicalSystem
    # AbstractSystem requirements
    eqs::Vector{Equation}
    states::Vector
    ps::Vector
    name::Symbol
    defaults::Dict
    # Extra fields
    n_dim::Int
    n_coll::Int
    n_mesh::Int
    t_coll::Vector{T}
    D::Matrix{T}
    D⁻¹::Matrix{T}
    ode_ps::Vector
    ode_u₀::Vector
    ode_u₁::Vector
    ode_t::Vector
end

function CollocationSystem(
    T::Type, sys::ODESystem, n_mesh, n_coll; name=gensym(nameof(sys)), free_ps=false
)
    # Turn the ODESystem into a function we can evaluate
    iv = MTK.get_iv(sys)  # independent variable
    sts = states(sys)
    ps = parameters(sys)
    f = @RuntimeGeneratedFunction(
        build_function([x.rhs for x in equations(sys)], sts, ps, iv)[1]
    )
    n_dim = length(sts)

    # Namespace the core symbols to avoid name collisions
    @variables coll₊u[1:n_mesh, 1:(n_coll + 1), 1:n_dim]
    for i in 1:(n_mesh - 1)
        coll₊u[i, end, :] .= coll₊u[i + 1, 1, :]  # continuity
    end
    @parameters coll₊ka[1:n_mesh]
    @variables coll₊t[0:1]

    # Regenerate the parameters as variables
    p = [Num(Variable(nameof(par))) for par in ps]

    # Generate the collocation equations for each mesh interval
    #   D*u ~ Tp*f(u, p, t)
    # where D is the differentiation matrix, and Tp is the time interval
    # the ka variables enable mesh interval scaling (assumes sum(ka) == 1)

    t_coll_full, D_full = cheb(T, n_coll)

    # Calculate the corresponding integration matrix (think about solving D*x = f(t) with appropriate boundary conditions)
    D_full⁻¹ = [zeros(1, n_coll + 1); zeros(n_coll, 1) inv(D_full[2:end, 2:end])]

    # End points aren't used in the collocation equations
    t_coll = t_coll_full[1:(end - 1)]
    D = D_full[1:(end - 1), :]

    # Interval length
    Tp = coll₊t[2] - coll₊t[1]

    # Collocation equations
    eqns = Vector{Equation}()
    for i in 1:n_mesh
        Dx = [D * coll₊u[i, :, k] for k in 1:n_dim]
        t_int =
            coll₊t[1] .+
            Tp .* (coll₊ka[i] .* t_coll .+ sum(coll₊ka[1:(i - 1)]; init=Num(0)))
        f_vals = [
            Tp * coll₊ka[i] * Base.invokelatest(f, coll₊u[i, j, :], p, t_int[j]) for
            j in eachindex(t_int)
        ]
        append!(
            eqns, [0 ~ Dx[k][j] - f_vals[j][k] for k in eachindex(Dx), j in eachindex(f_vals)]
        )
    end

    # Bindings to the inner variables
    sts_names = nameof.(MTK.operation.(sts))
    sts_begin = [Num(Variable(st, 0)) for st in sts_names]
    append!(eqns, 0 .~ coll₊u[1, 1, :] .- sts_begin)
    sts_end = [Num(Variable(st, 1)) for st in sts_names]
    append!(eqns, 0 .~ coll₊u[end, end, :] .- sts_end)

    # Defaults
    defaults = merge(
        MTK.get_defaults(sys), Dict(MTK.value(ka) => one(T) / n_mesh for ka in coll₊ka)
    )

    new_sts = [sts_begin; sts_end; unique(vec(coll₊u)); coll₊t]
    new_ps = copy(coll₊ka)
    if free_ps
        append!(new_sts, p)
    else
        append!(new_ps, p)
    end

    return CollocationSystem(
        eqns,  # eqs
        MTK.value.(new_sts),  # states
        MTK.value.(new_ps),  # ps
        name,  # name
        defaults,  # defaults
        n_dim,  # n_dim
        n_coll,  # n_coll
        n_mesh,  # n_mesh
        t_coll_full,  # t_coll
        D_full,  # D
        D_full⁻¹,  # D⁻¹
        p,  # ode_ps
        sts_begin,  # ode_u₀
        sts_end,  # ode_u₁
        coll₊t,  # ode_t
    )
end

function CollocationSystem(sys::ODESystem, args...; kwargs...)
    return CollocationSystem(Float64, sys, args...; kwargs...)
end

"""
    $SIGNATURES

Add equations that are a function of the boundary states and times to `coll`. The input `bc`
is assumed to be a function of the form `(u₀, t₀, u₁, t₁, p) -> equation` where `u₀` and
`u₁` are state variables at the start and end of the interval respectively, `t₀` and `t₁`
are the start and end times, and `p` are the underlying ODE system parameters.

The value returned from `bc` can be an equation or a vector of equations.

`u₀` and `u₁` have the states in the same order as the underlying ODE system.

## Example - periodic boundary conditions

```julia
add_boundarycondition!(coll, (u₀, t₀, u₁, t₁, p) -> u₀ .~ u₁)
```

Note that `.~` is used since `u₀` and `u₁` are vectors of variables.
"""
function add_boundarycondition!(coll::CollocationSystem, bc)
    eqn = bc(coll.ode_u₀, coll.ode_t[1], coll.ode_u₁, coll.ode_t[2], coll.ode_ps)
    if eqn isa Equation
        push!(coll.eqs, eqn)
    elseif eqn isa Vector{Equation}
        append!(coll.eqs, eqn)
    elseif eqn isa Num
        push!(coll.eqs, 0 ~ eqn)
    elseif eqn isa Vector{Num}
        append!(coll.eqs, 0 .~ eqn)
    else
        throw(ArgumentError("Input function should return a (vector of) Equation or Num"))
    end
    return coll
end

# CollocationSystem does not allow embedded systems (it's not meaningful)
ModelingToolkit.get_systems(::CollocationSystem) = NonlinearSystem[]

# Convert a CollocationSystem to a NonlinearSystem
function ModelingToolkit.NonlinearSystem(coll::CollocationSystem)
    return NonlinearSystem(
        MTK.get_eqs(coll),
        MTK.get_states(coll),
        MTK.get_ps(coll);
        name=nameof(coll),
        defaults=MTK.get_defaults(coll),
    )
end
