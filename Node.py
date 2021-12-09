
class Node:
    def __init__(self, label):
        self.label = label
        self.edges = []

    def addEdge(self, to):
        if to not in self.edges and to.label != self.label:
            self.edges.append(to)

    def print(self):
        print("Node", self.label)
        for edge in self.edges:
            print("->", edge.label)

    def nodesConNotInc(self, orig_weights, visited_arr, before = ""):
        visited_arr[orig_weights.index(self.label)] = True

        nodes = [self]
        for node in self.edges:
            visited = visited_arr[orig_weights.index(node.label)]
            if not visited:
                visited_arr[orig_weights.index(node.label)] = True
                connected_to_this_node, visited_arr = node.nodesConNotInc(orig_weights, visited_arr, before + "   ")
                for n in connected_to_this_node:
                    if n not in nodes:
                        nodes.append(n)
                        
        return nodes, visited_arr