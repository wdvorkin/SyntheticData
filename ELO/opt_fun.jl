function OPF(net,d)
    # original DC-OPF problem
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))
    # set_optimizer_attribute(model, "OutputFlag", 0)
    # model variables
    @variable(model, p[1:net[:N]])
    # model objective
    @objective(model, Min, net[:c1]'p)
    # OPF equations
    @constraint(model, λ, ones(net[:N])'*(p .- d) .>= 0)
    @constraint(model, μ̅, net[:f̅] .>=   net[:F]*(p .- d))
    @constraint(model, μ̲, net[:f̅] .>= - net[:F]*(p .- d))
    @constraint(model, γ̅,  -p .>= -net[:p̅])
    @constraint(model, γ̲,  -net[:p̲] .>= -p)
    # solve model
    optimize!(model)
    status = "$(termination_status(model))"
    if status == "OPTIMAL"
        sol = Dict(:status => "$(termination_status(model))",
        :obj => JuMP.objective_value(model),
        :p => JuMP.value.(p),
        :CPUtime => solve_time(model)
        )
    else
        sol = Dict(:status => termination_status(model))
    end
    return sol
end


function OPF_R(net,d)
    # relaxed DC-OPF problem
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))
    # set_optimizer_attribute(model, "OutputFlag", 0)
    # model variables
    @variable(model, p[1:net[:N]])
    @variable(model, v[1:net[:E]])
    # model objective
    @objective(model, Min, net[:c1]'p + set[:ψ]*sum(v))
    # OPF equations
    @constraint(model, λ, ones(net[:N])'*(p .- d) .>= 0)
    @constraint(model, μ̅, - net[:F]*(p .- d) .>= - net[:f̅] - v)
    @constraint(model, μ̲, net[:f̅] + v .>= - net[:F]*(p .- d))
    @constraint(model, γ̅,  -p .>= -net[:p̅])
    @constraint(model, γ̲,  -net[:p̲] .>= -p)
    @constraint(model, κ,  0 .>= -v)
    # solve model
    optimize!(model)
    status = "$(termination_status(model))"
    if status == "OPTIMAL"
        sol = Dict(:status => termination_status(model),
        :obj => JuMP.objective_value(model),
        :p => JuMP.value.(p),
        :v => JuMP.value.(v),
        :CPUtime => solve_time(model)
        )
    else
        sol = Dict(:status => termination_status(model))
    end
    return sol
end

function OPF_PP(net,d_o,OPF_records,wc_cost,k_t)
    t = length(k_t)
    model = Model(optimizer_with_attributes(Gurobi.Optimizer))
    set_optimizer_attribute(model, "OutputFlag", 0)
    # upper-level variables 
    @variable(model, d[1:net[:N]]>=0)
    @variable(model, r_d)
    @variable(model, r_c[1:t])
    # primal lower-level variables
    @variable(model, p[1:net[:N],1:t])
    # @variable(model, cost[1:t])
    # dual variables
    @variable(model, λ[1:t])
    @variable(model, μ̅[1:net[:E],1:t]>=0)
    @variable(model, μ̲[1:net[:E],1:t]>=0)
    @variable(model, γ̅[1:net[:N],1:t]>=0)
    @variable(model, γ̲[1:net[:N],1:t]>=0)
    # objective function
    @objective(model, Min, 1 * r_d +  sum(r_c[τ] for τ in 1:t))
    # objective aux constraints 
    @constraint(model, [r_d;d .- d_o] in SecondOrderCone())
    @constraint(model, aux_con[τ=1:t], [r_c[τ];wc_cost[τ] .- OPF_records[k_t[τ]][:c1]'p[:,τ]] in SecondOrderCone())
    for τ in 1:t
        # primal constriants 
        @constraint(model, ones(net[:N])'*(p[:,τ] .- d) .== 0)
        @constraint(model, - net[:F]*(p[:,τ] .- d) .>= - OPF_records[k_t[τ]][:f̅])
        @constraint(model, OPF_records[k_t[τ]][:f̅] .>= - net[:F]*(p[:,τ] .- d))
        @constraint(model, -p[:,τ] .>= -OPF_records[k_t[τ]][:p̅])
        @constraint(model, -OPF_records[k_t[τ]][:p̲] .>= -p[:,τ])
        # dual constraints
        @constraint(model, λ[τ] * ones(net[:N]) - net[:F]' * μ̅[:,τ] + net[:F]' * μ̲[:,τ] - γ̅[:,τ] + γ̲[:,τ] .<= OPF_records[k_t[τ]][:c1])
        # # strong duality 
        # complementarity conditions formulated as SOS1 constraints
        for i in 1:net[:E]
            @constraint(model, [μ̅[i,τ] ; OPF_records[k_t[τ]][:f̅][i] .- net[:F][i,:]'*(p[:,τ] .- d)] in SOS1())
            @constraint(model, [μ̲[i,τ] ; net[:F][i,:]'*(p[:,τ] .- d) .+ OPF_records[k_t[τ]][:f̅][i]] in SOS1())
        end
        for i in 1:net[:N]
            @constraint(model, [γ̅[i,τ] ; OPF_records[k_t[τ]][:p̅][i] .- p[i,τ]] in SOS1())
            @constraint(model, [γ̲[i,τ] ; p[i,τ] .- OPF_records[k_t[τ]][:p̲][i]] in SOS1())
        end
    end


    @constraint(model, d[findall(net[:d] .== 0.0)] .== 0 )

    # solve primal-dual model
    optimize!(model)
    @info("done solving the PP problem: $(termination_status(model))")
    status = "$(termination_status(model))"
    if status == "OPTIMAL"
        sol = Dict(:status => termination_status(model),
        :obj => JuMP.objective_value(model),
        :p => JuMP.value.(p),
        :CPUtime => solve_time(model),
        :d => JuMP.value.(d),
        :r_d => JuMP.value.(r_d),
        :r_c => JuMP.value.(r_c)
        )
    else
        sol = Dict(:status => termination_status(model))
    end
end

function run_TCO(net,set,OPF_records,φ̅_o,ξ,T)
    @info("starting the TCO algorithm for T=$(T)")
    φ̅ = zeros(net[:E],T)
    k_t = []; wc_cost = []
    for t in 1:T
        # Step 2: run EM to find the w.c. query
        ΔC = zeros(set[:S]); ΔC̃ = zeros(set[:S])
        @showprogress 1 "Running EM at iteration $(t)" for i in 1:set[:S]
            sol_opf = OPF(OPF_records[i],net[:f̅])
            if t == 1
                sol_opf_rel = OPF_R(OPF_records[i],φ̅_o)
            else
                sol_opf_rel = OPF_R(OPF_records[i],φ̅[:,t-1]) 
            end
            sol_opf[:status] == "OPTIMAL " ? ΔC[i] = norm(sol_opf[:obj] - sol_opf_rel[:obj],1) : ΔC[i] = 0
            lap_scale_EM = set[:α]*set[:c_max]/(set[:ε]/(4*T))
            ΔC̃[i] = ΔC[i] + ξ[:ξ_EM][i,t]*lap_scale_EM
        end
        ind_max = findall(x->x==maximum(ΔC̃),ΔC̃)[1]
        push!(k_t,ind_max)
        # Step 3: LM to find the w.c. OPF cost 
        sol_opf = OPF(OPF_records[k_t[1]],net[:f̅])
        lap_scale_LM = set[:α]*set[:c_max]/set[:ε]/(4*T)
        push!(wc_cost, sol_opf[:obj] .+ ξ[:ξ_LM][t]*lap_scale_LM)
        # Step 4: post-processing 
        @show wc_cost, k_t
        if t == 1
            sol_pp = OPF_PP(net,φ̅_o,OPF_records,wc_cost,k_t)
        else
            sol_pp = OPF_PP(net,φ̅[:,t-1],OPF_records,wc_cost,k_t) 
        end
        
        φ̅[:,t] .= sol_pp[:φ]
    end
    feas = opf_feas_test(OPF_records,set,φ̅[:,T])
    return Dict(:φ̅ => φ̅[:,T], :feas => feas)
end