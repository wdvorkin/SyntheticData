using PowerModels
using Statistics, LinearAlgebra, Distributions, Random
using JuMP, Gurobi, MosekTools, Mosek
using DataFrames, CSV, Tables, Plots
using ProgressMeter, ArgParse

include("aux_fun.jl")
include("opt_fun.jl")

# Load network data using PowerModels
cd(dirname(@__FILE__))
PowerModels.silence()
caseID="data/pglib_opf_case73_ieee_rts.m" 
net = load_network_data(caseID)
net[:f̅] .= net[:f̅] .* 0.6

# parse arguments 
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--iterlim", "-t"
            help = "iteration limit"
            arg_type = Int64
            default = 5
        "--num_record", "-s"
            help = "number of opf records"
            arg_type = Int64
            default = 100
        "--adjacency", "-a"
            help = "adjacency of opf datasets"
            arg_type = Float64
            default = 15.0
        "--loss", "-e"
            help = "privacy loss ε"
            arg_type = Float64
            default = 1.0
        "--num_runs", "-r"
            help = "numer of runs of the algorithm"
            arg_type = Int64
            default = 1
    end
    return parse_args(s)
end
args = parse_commandline()

# experiment settings
set = Dict(:ψ => 3000, :α => args["adjacency"], :ε => args["loss"], :T => args["iterlim"], :S => args["num_record"], :c_max => 0.0)

# load OPF records data
OPF_records = create_OPF_records(set)

# find the max cost coefficient in the database:
set[:c_max] = maximum([maximum(OPF_records[i][:c1][net[:gen_bus_ind]]) for i in 1:set[:S]])

Random.seed!(1)
# Initialize synthetic dataset 
d_o = max.(0,net[:d] .+ rand(Laplace(0,set[:α]/(set[:ε]/2)),net[:N]));
d_o[findall(net[:d].==0.0)] .= 0 

# sample random noise for Exponential and Laplace mechanisms
ξ_EM = rand(Laplace(0,1),set[:S],set[:T])
ξ_LM = rand(Laplace(0,1),set[:T])
ξ = Dict(:ξ_EM => ξ_EM, :ξ_LM => ξ_LM)

@info("starting the ELO algorithm for T=$(set[:T])")
d = zeros(net[:N],set[:T])
k_t = []; wc_cost = []
for t in 1:set[:T]
    # Step 2: run EM to find the w.c. query
    ΔC = zeros(set[:S]); ΔC̃ = zeros(set[:S])
    @showprogress 1 "Running EM at iteration $(t)" for i in 1:set[:S]
        sol_opf = OPF(OPF_records[i],net[:d])
        if t == 1
            sol_opf_rel = OPF_R(OPF_records[i],d_o)
        else
            sol_opf_rel = OPF_R(OPF_records[i],d[:,t-1]) 
        end
        ΔC[i] = norm(sol_opf[:obj] - sol_opf_rel[:obj],1)
        lap_scale_EM = set[:α]/(set[:ε]/(4*set[:T]))
        ΔC̃[i] = ΔC[i] + ξ[:ξ_EM][i,t]*lap_scale_EM
    end
    ind_max = findall(x->x==maximum(ΔC̃),ΔC̃)[1]
    push!(k_t,ind_max)
    @show k_t
    # Step 3: LM to find the w.c. OPF cost 
    sol_opf = OPF(OPF_records[k_t[1]],net[:d])
    lap_scale_LM = set[:α]*set[:c_max]/set[:ε]/(4*set[:T])
    push!(wc_cost, sol_opf[:obj] .+ ξ[:ξ_LM][t]*lap_scale_LM)
    # Step 4: post-processing 
    @show wc_cost, k_t
    if t == 1 
        sol_pp = OPF_PP(net,d_o,OPF_records,wc_cost,k_t)
    else
        sol_pp = OPF_PP(net,d[:,t-1],OPF_records,wc_cost,k_t)
    end
    d[:,t] .= sol_pp[:d]
end

anim = @animate for t in 1:set[:T]
    @show t
    feas, subopt_ = opf_feas_opt_test(OPF_records,set,d[:,t])
    plot(legend=:topleft,title="iteration: $t   infeas: $(round(feas,digits=1))%     suboptimality: $(round(subopt_,digits=1))%")
    bar!(1:net[:D],[net[:d][sortperm(net[:d])][end-net[:D]+1:end]],fillalpha=0.5,label="real load",color=:blue)
    bar!(1:net[:D],d[sortperm(net[:d])[end-net[:D]+1:end],t],fillalpha=0.5,label="synthetic load",color=:green)
    plot!(frame=:box)
    plot!(xlabel="load index (sorted)")
    plot!(ytickfont=font(15))
    plot!(xtickfont=font(15))
    plot!(ylabelfontsize=15)
    plot!(xlabelfontsize=15)
    plot!(titlefont=font(15))
    plot!(legendfontsize=12)
    plot!(xlims=(1,net[:D]))
    plot!(ylims=(0,400))
    plot!(size=(700,700))
end

gif(anim, "animation_ELO.gif", fps = 2)
