import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
	gex = nx.Graph()
	gex.add_nodes_from(range(100, 110))
	#plt.interactive(True)
	nx.draw(gex)
	plt.show()

