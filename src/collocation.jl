function cheb(T::Type, N::Integer)
    if N == 0
        return (ones(T, 1), zeros(T, 1, 1))
    else
        x = (-cos.((T(π) * (0:N)) ./ N) .+ 1) ./ 2
        c = [2; ones(T, N - 1, 1); 2] .* (-1) .^ (0:N)
        dx = x .- x'
        D = (c * (1 ./ c)') ./ (dx + I)
        D[diagind(D)] .-= vec(sum(D; dims = 2))
        return (x, D)
    end
end

function collocation(T::Type, sys::ODESystem, n_mesh::Integer, n_coll::Integer; name)
    # Turn the ODESystem into a function we can evaluate
    iv = get_iv(sys)
    sts = states(sys)
    ps = parameters(sys)
    f = @RuntimeGeneratedFunction(
        build_function([x.rhs for x in equations(sys)], sts, ps, iv)[1]
    )
    n_var = length(sts)

    # Generate the inner symbols
    @variables u[1:n_mesh, 1:(n_coll+1), 1:n_var]
    @parameters ka[1:n_mesh]
    for i = 1:(n_mesh-1)
        u[i, end, :] .= u[i+1, 1, :]  # continuity
    end
    @named coll = NonlinearSystem([], vec(u), ka)

    # Namespace the inner symbols
    @variables coll₊u[1:n_mesh, 1:(n_coll+1), 1:n_var]
    for i = 1:(n_mesh-1)
        coll₊u[i, end, :] .= coll₊u[i+1, 1, :]  # continuity
    end
    @parameters coll₊ka[1:n_mesh]

    # Regenerate the parameters as variables
    p = [Num(Variable(nameof(par))) for par in ps]

    # Start/end times
    @variables t[0:1]

    # Generate the collocation equations for each mesh interval
    #   D*u ~ Tp*f(u, p, t)
    # where D is the differentiation matrix, and Tp is the time interval
    # the ka variables enable mesh interval scaling (assumes sum(ka) == 1)

    t_int_base, D = cheb(T, n_coll)
    # End points aren't used in the collocation equations (they are used to enforce continuity instead)
    t_int_base = t_int_base[1:(end-1)]
    D = D[1:(end-1), :]

    Tp = t[2] - t[1]

    # Collocation equations
    eqns = Vector{Equation}()
    for i = 1:n_mesh
        Dx = [D * coll₊u[i, :, k] for k = 1:n_var]
        t_int = Tp * (coll₊ka[i] .* t_int_base .+ sum(coll₊ka[1:(i-1)]; init = Num(0)))
        f_vals = [
            Tp * coll₊ka[i] * Base.invokelatest(f, coll₊u[i, j, :], p, t_int[j]) for
            j in eachindex(t_int)
        ]
        append!(
            eqns,
            [Dx[k][j] ~ f_vals[j][k] for k in eachindex(Dx), j in eachindex(f_vals)],
        )
    end

    # Bindings to the inner variables
    sts_names = nameof.(operation.(sts))
    sts_begin = [Num(Variable(st, 0)) for st in sts_names]
    append!(eqns, coll₊u[1, 1, :] .~ sts_begin)
    sts_end = [Num(Variable(st, 1)) for st in sts_names]
    append!(eqns, coll₊u[end, end, :] .~ sts_end)

    return NonlinearSystem(eqns, [sts_begin; sts_end; p; t], []; name, systems = [coll])
end

function collocation(sys::ODESystem, args...; kwargs...)
    return collocation(Float64, sys, args...; kwargs...)
end

function testsystem()
    @parameters t a b
    @variables x(t) x′(t)
    D = Differential(t)
    return ODESystem([D(x) ~ x′, D(x′) ~ -a * x′ - x - b * x^3])
end
