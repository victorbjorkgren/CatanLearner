import networkx as nx

def node_dist(x, y, cx, cy):
    """Distance of each node from the center of the innermost layer"""
    return abs(cx - x) + abs(cy - y)


def remove_unwanted_nodes(G, m):
    """Remove all the nodes that don't belong to an m-layer hexagonal ring."""

    # Compute center of all the hexagonal rings as cx, cy
    cx, cy = m - 0.5, 2 * m - (m % 2)  # odd is 2m-1, even is 2m

    # in essence, we are converting from a rectangular grid to a hexagonal ring... based on distance.
    unwanted = []
    for n in G.nodes:
        x, y = n
        # keep short distance nodes, add far away nodes to the list called unwanted
        if node_dist(x, y, cx, cy) > 2 * m:
            unwanted.append(n)

    # now we are removing the nodes from the Graph
    for n in unwanted:
        G.remove_node(n)

    return G


def make_hex_grid(m):
    # change m here. 1 = 1 layer, single hexagon.
    G = nx.hexagonal_lattice_graph(2 * m - 1, 2 * m - 1, periodic=False,
                                   # with_positions=True,
                                   create_using=None)
    # pos = nx.get_node_attributes(G, 'pos')
    G = remove_unwanted_nodes(G, m)

    # render the result
    # plt.figure(figsize=(9, 9))
    # nx.draw(G, pos=pos, with_labels=True)
    # plt.axis('scaled')
    # plt.show()

    return G
