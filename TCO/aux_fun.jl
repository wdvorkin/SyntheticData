# auxiliary functions
ns(l) = Int(net[:n_s][l])
nr(l) = Int(net[:n_r][l])
Φ(x) = quantile(Normal(0,1),1-x)
function remove_col_and_row(B,refbus)
    @assert size(B,1) == size(B,2)
    n = size(B,1)
    return B[1:n .!= refbus, 1:n .!= refbus]
end
function build_B̆(B̂inv,refbus)
    Nb = size(B̂inv,1)+1
    B̆ = zeros(Nb,Nb)
    for i in 1:Nb, j in 1:Nb
        if i < refbus && j < refbus
            B̆[i,j] = B̂inv[i,j]
        end
        if i > refbus && j > refbus
            B̆[i,j] = B̂inv[i-1,j-1]
        end
        if i > refbus && j < refbus
            B̆[i,j] = B̂inv[i-1,j]
        end
        if i < refbus && j > refbus
            B̆[i,j] = B̂inv[i,j-1]
        end
    end
    return B̆
end
function load_network_data(caseID)
    data_net = PowerModels.parse_file(caseID)
    # Network size
    G = length(data_net["gen"])
    N = length(data_net["bus"])
    E = length(data_net["branch"])
    D = length(data_net["load"])

    # order bus indexing
    bus_keys=collect(keys(data_net["bus"]))
    bus_key_dict = Dict()
    for i in 1:N
        push!(bus_key_dict, i => bus_keys[i])
    end
    node(key) = [k for (k,v) in bus_key_dict if v == key][1]

    # Load generation data
    gen_key=collect(keys(data_net["gen"]))
    p̅ = zeros(G); p̲ = zeros(G); c1 = zeros(G); c2 = zeros(G); M_p = zeros(N,G)
    for g in gen_key
        p̅[parse(Int64,g)] = data_net["gen"][g]["pmax"]*data_net["baseMVA"]
        # p̲[parse(Int64,g)] = data_net["gen"][g]["pmin"]*data_net["baseMVA"]
        p̲[parse(Int64,g)] = 0
        if sum(data_net["gen"][g]["cost"]) != 0
            c1[parse(Int64,g)] = data_net["gen"][g]["cost"][1] / data_net["baseMVA"]
            c2[parse(Int64,g)] = data_net["gen"][g]["cost"][2] / data_net["baseMVA"]^2
        end
        M_p[node(string(data_net["gen"][g]["gen_bus"])),parse(Int64,g)] = 1
    end
    # sum(c2) == 0 ? c2 = 0.05*c1 : NaN

    # Load demand data
    load_key=collect(keys(data_net["load"]))
    d = zeros(D); M_d = zeros(N,D)
    for h in load_key
        d[parse(Int64,h)] = data_net["load"][h]["pd"]*data_net["baseMVA"] + 1e-3
        M_d[node(string(data_net["load"][h]["load_bus"])),parse(Int64,h)] = 1
    end

    # Load transmission data
    line_key=collect(keys(data_net["branch"]))
    β = zeros(E); f̅ = zeros(E); n_s = trunc.(Int64,zeros(E)); n_r = trunc.(Int64,zeros(E))
    for l in line_key
        β[data_net["branch"][l]["index"]] = -imag(1/(data_net["branch"][l]["br_r"] + data_net["branch"][l]["br_x"]im))
        n_s[data_net["branch"][l]["index"]] = data_net["branch"][l]["f_bus"]
        n_r[data_net["branch"][l]["index"]] = data_net["branch"][l]["t_bus"]
        f̅[data_net["branch"][l]["index"]] = data_net["branch"][l]["rate_a"]*data_net["baseMVA"]
    end
    # merge parallel lines
    ff = zeros(N,N); ββ = zeros(N,N)
    for l in line_key
        ff[node(string(n_s[data_net["branch"][l]["index"]])),node(string(n_r[data_net["branch"][l]["index"]]))] += f̅[data_net["branch"][l]["index"]]
        ff[node(string(n_r[data_net["branch"][l]["index"]])),node(string(n_s[data_net["branch"][l]["index"]]))] += f̅[data_net["branch"][l]["index"]]
        ββ[node(string(n_s[data_net["branch"][l]["index"]])),node(string(n_r[data_net["branch"][l]["index"]]))]  = β[data_net["branch"][l]["index"]]
        ββ[node(string(n_r[data_net["branch"][l]["index"]])),node(string(n_s[data_net["branch"][l]["index"]]))]  = β[data_net["branch"][l]["index"]]
    end
    # find all parallel lines
    parallel_lines = []
    for l in line_key, e in line_key
        if l != e && node(string(n_s[data_net["branch"][l]["index"]])) == node(string(n_s[data_net["branch"][e]["index"]])) && node(string(n_r[data_net["branch"][l]["index"]])) == node(string(n_r[data_net["branch"][e]["index"]]))
            push!(parallel_lines,l)
        end
    end
    # for l in sort!(parallel_lines)
    #     println("$(l) ... $(data_net["branch"][l]["f_bus"]) ... $(data_net["branch"][l]["t_bus"]) ... $(f̅[data_net["branch"][l]["index"]]) ... $(β[data_net["branch"][l]["index"]])")
    # end
    # update number of edges
    E = E - Int(length(parallel_lines)/2)
    # get s and r ends of all edge
    n_s = trunc.(Int64,zeros(E)); n_r = trunc.(Int64,zeros(E))
    ff = LowerTriangular(ff)
    for l in 1:E
        n_s[l] = findall(!iszero, ff)[l][1]
        n_r[l] = findall(!iszero, ff)[l][2]
    end
    β = zeros(E); f̅ = zeros(E);
    for l in 1:E
        β[l] = ββ[n_s[l],n_r[l]]
        f̅[l] = ff[n_s[l],n_r[l]]
    end

    # Find reference node
    ref = 0
    for n in 1:N
        if sum(M_p[n,:]) == 0 &&  sum(M_d[n,:]) == 0 == 0
            ref = n
        end
    end

    # Compute PTDF matrix
    B_line = zeros(E,N); B̃_bus = zeros(N,N); B = zeros(N,N)
    for n in 1:N
        for l in 1:E
            if n_s[l] == n
                B[n,n] += β[l]
                B_line[l,n] = β[l]
            end
            if n_r[l] == n
                B[n,n] += β[l]
                B_line[l,n] = -β[l]
            end
        end
    end
    for l in 1:E
        B[Int(n_s[l]),Int(n_r[l])] = - β[l]
        B[Int(n_r[l]),Int(n_s[l])] = - β[l]
    end
    B̃_bus = remove_col_and_row(B,ref)
    B̃_bus = inv(B̃_bus)
    B̃_bus = build_B̆(B̃_bus,ref)
    PTDF = B_line*B̃_bus

    # safe network data
    net = Dict(
    # transmission data
    :f̅ => f̅, :n_s => n_s, :n_r => n_r, :F => round.(PTDF,digits=8),
    # load data
    :d => round.(M_d*d,digits=5),
    # generation data
    :p̅ => round.(M_p*p̅,digits=5), :p̲ => round.(M_p*p̲,digits=5),
    :c1 => round.(M_p*c1,digits=5), :gen_bus_ind => findall(!iszero,vec(sum(M_p,dims=2))),
    # graph data
    :N => N, :E => E, :G => G, :D => D, :ref => ref
    )
    return net
end
function create_OPF_records(set)
    OPF_δ = Dict(:c => CSV.File("data/OPF_records_c.csv",header=0) |> Tables.matrix,
                 :d => CSV.File("data/OPF_records_d.csv",header=0) |> Tables.matrix, 
                 :p => CSV.File("data/OPF_records_p.csv",header=0) |> Tables.matrix)
    # OPF_records
    OPF_records = Dict()
    for i in 1:set[:S]
        network = deepcopy(net)
        network[:c1] = OPF_δ[:c][:,i]
        network[:p̅] = net[:p̅] .* OPF_δ[:p][:,i]
        network[:d] = net[:d] .* OPF_δ[:d][:,i]
        OPF_records[i] = network
    end
    return OPF_records
end
function opf_feas_opt_test(OPF_records,set,φ̅)
    inf_coutner = [0]
    sub_opt = zeros(set[:S])
    for i in 1:set[:S]
        sol_real = OPF(OPF_records[i],net[:f̅])

        sol_opf = OPF(OPF_records[i],φ̅)
        sol_opf[:status] != "OPTIMAL" ? push!(inf_coutner,1) : NaN

        sol_opf = OPF_R(OPF_records[i],φ̅)

        sub_opt[i] = norm(sol_real[:obj] .- sol_opf[:obj],1)/sol_real[:obj]*100
    end
    return sum(inf_coutner)/set[:S]*100, mean(sub_opt)
end