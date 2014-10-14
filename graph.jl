module Graph

export Attributes, Graph, add_vertex!, add_edge!, set_all_node_attributes!
export make_cycle
export apsp_floyd_warshall, commute_time, stress_majorization

import JSON

################################################################################
# A graph will be a dict{Int32, (String, Array{(Int32, Dict{String, Any})})}

typealias Attributes Dict{ASCIIString, Any}
typealias DirectedGraph Dict{Int32, (Attributes, Array{(Int32, Attributes)})}

function add_vertex!(g, from, to_list)
    push!(g, from, (["name" => string(from)], [(i, Attributes()) for i in to_list]))
end

function add_edge!(g, from, to; attrs=None)
    if attrs == None
        attrs = Attributes()
    end
    push!(g[from][2], (to, attrs))
end

function set_all_node_attributes!(g, name, value)
    @assert length(value) == length(g)
    for (from, (node_attrs, to_list)) in g
        node_attrs[name] = value[from]
    end
end

function for_all_edges(g, f)
    for (from, (node_attrs, to_list)) in g
        for (to, edge_attrs) in to_list
            f(from, to, edge_attrs)
        end
    end
end

function for_all_nodes(g, f)
    for (from, (node_attrs, to_list)) in g
        f(from, node_attrs, to_list)
    end
end

function graph_to_graphviz(g; f=STDOUT, directed=false)
    reserved_edge_attrs = ["name", "x", "y", "pos"]

    @printf(f, "graph G {\n")
    for_all_nodes(g, (node, attrs, edge_list) -> begin
        @printf(f, "  %s [label=\"%s\"", node, attrs["name"])
        if haskey(attrs, "pos")
            @printf(f, ", pos=%s", string(attrs["pos"]))
        elseif haskey(attrs, "x") && haskey(attrs, "y") 
            @printf(f, ", pos=\"%s, %s\"", attrs["x"], attrs["y"])
        end
        for (key, value) in attrs
            r = find(x -> x == key, reserved_edge_attrs)
            if length(r) == 0
                @printf(f, ", %s=\"%s\"", key, value)
            end
        end
        @printf(f, "]\n")
    end)

    if !directed
        for_all_edges(g, (from, to, edge_attrs) -> begin
            if to > from
                @printf(f, "  %s -- %s", from, to)
                if length(edge_attrs) > 0
                    @printf(f, " [")
                    for (i, (name, key)) in enumerate(edge_attrs)
                        @printf(f, "%s%s=\"%s\"", (i == 1 ? "" : ", "), name, key)
                    end
                    @printf(f, "]")
                end
                @printf(f, "\n")
            end
        end)
    else
        for_all_edges(g, (from, to, edge_attrs) -> begin
            @printf(f, "  %s -> %s", from, to)
            if length(edge_attrs) > 0
                @printf(f, " [")
                for (i, (name, key)) in enumerate(edge_attrs)
                    @printf(f, "%s%s=\"%s\"", (i == 1 ? "" : ", "), name, key)
                end
                @printf(f, "]")
            end
            @printf(f, "\n")
        end)
    end
    @printf(f, "}\n")
end

function graph_to_json(g)

    function edges()
        result = Array(Attributes, 0)
        for_all_edges(g, (from, to, edge_attrs) -> begin
            t = (ASCIIString=>Any)["source" => from, "target" => to]
            push!(result, t)
        end)
        result
    end
    
    JSON.json(["name"  => "G",
               "type"  => "digraph",
               "nodes" => [ node_attrs for (from, (node_attrs, to_list)) in g ],
               "edges" => edges()])
end

################################################################################
# some standard graphs

function make_cycle(n)
    result = DirectedGraph()
    for i = 1:n
        ii = i - 1
        add_vertex!(result, i, [1 + ((i+n-2) % n), 1 + i % n])
    end
    result
end

################################################################################
# some standard metrics

function apsp_floyd_warshall(g)
    n = length(g)
    result = Array(Float64, n, n)
    fill!(result, 1000000.0)
    for i in 1:n
        result[i,i] = 0
    end
    for (from, (node_attrs, to_list)) in g
        for (to, edge_attrs) in to_list
            if to > from
                weight = if haskey(edge_attrs, "weight") edge_attrs["weight"] else 1.0 end
                result[from, to] = weight
                result[to, from] = weight
            end
        end
    end
    for i in 1:n
        for j in 1:n
            for k in 1:n
                result[j,k] = min(result[j,i] + result[i,k], result[j,k])
            end
        end
    end
    result
end

function commute_time(g)
    n = length(g)
    m = Array(Float64, n, n)
    fill!(m, 0)
    volume = 0.0
    for (from, (node_attrs, to_list)) in g
        for (to, edge_attrs) in to_list
            if to > from
                weight = if haskey(edge_attrs, "weight") edge_attrs["weight"] else 1.0 end
                m[from, to[1]] -= weight
                m[to[1], from] -= weight
                m[from, from] += weight
                m[to[1], to[1]] += weight
                volume += weight
            end
        end
    end
    inner_products = moore_penrose_pseudo_inverse(m)
    result = Array(Float64, n, n)
    for i in 1:n
        for j in 1:n
            v = (inner_products[i,i] + inner_products[j,j] - 2.0 * inner_products[i,j])
            if v < 0
                v = -v
            end
            result[i,j] = (volume * v) ^ 0.5
        end
    end
    result
end

################################################################################

function stress_majorization(g; metric = apsp_floyd_warshall, dim = 2, alpha = 2, max_iter = 100, verbose = false)
    n = length(g)
    d = metric(g)
    w = map(x -> if x == 0 0.0 else 1.0/(x ^ alpha) end, d)
    delta = Array(Float32, n, n)
    for i in 1:n
        for j in 1:n
            delta[i,j] = d[i,j] == 0.0 ? 0.0 : (d[i,j] * w[i,j])
        end
    end

    w = make_laplacian!(-w)

    # This could be made faster by avoiding half of the O(n^3) work
    L_pseudo_inv = moore_penrose_pseudo_inverse(w)

    result = rand(dim, n)

    function dist(v1, v2)
        d = sum((v1 - v2) .^ 2)
        if (d > 0)
            d ^ -0.5
        else
            0
        end
    end

    relative_movement = 1
    iter = 0
    while relative_movement > 1e-4 && iter < max_iter
        Lz = make_laplacian!(Float32[ - delta[i,j] * dist(result[:,i], result[:,j]) for i in 1:n, j in 1:n ])
        new_result = (result * Lz) * L_pseudo_inv 
        tl = sum(result .^ 2) # total_length
        tdl = sum((result - new_result) .^ 2) # total_delta_length
        relative_movement = (tdl / tl) ^ 0.5 / (n * dim)
        iter += 1
        if verbose
            println("relative change: $relative_movement, iteration $iter")
        end
        result = new_result
    end
    result
end

################################################################################

function moore_penrose_pseudo_inverse(m; d = 1e-3)
    function pseudo_inv_diag(v, d)
        [if x > d 1/x else x end for x in v]
    end
    svd = svdfact(m)
    spectral_norm = sum(svd[:S].^2)^0.5
    threshold = d * spectral_norm
    svd[:Vt]' * diagm(pseudo_inv_diag(svd[:S], threshold)) * svd[:U]'
end

function make_laplacian!(m)
    n = size(m, 1)
    for i in 1:n
        m[i, i] = 0.0
    end
        
    for i in 1:n
        for j in 1:n
            if i != j
                m[i, i] -= m[i, j]
            end
        end
    end
    m
end

end


