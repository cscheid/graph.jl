module Neato

import Graph

function neato!(g; scale=72, output="", args...)
    layout = Graph.stress_majorization(g; args...)
    Graph.set_all_node_attributes!(g, "x", layout[1,:] * scale)
    Graph.set_all_node_attributes!(g, "y", layout[2,:] * scale)
    if output != ""
        f = open(output, "w")
        Graph.graph_to_graphviz(g, f=f)
        close(f)
    end
    g
end

end
