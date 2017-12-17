import pickle
import networkx as nx

def exportX():
	dg = nx.DiGraph()
	dg.add_edge('a','b')
	dg.add_edge('a','c')
	out_file = open('/tmp/graph.txt', 'w')
	# Pickle the list using the highest protocol available: -1 or pickle.HIGHEST_PROTOCOL
	pickle.dump(dg, out_file , pickle.HIGHEST_PROTOCOL)


def importX():
	dg = pickle.load(open('/tmp/graph.txt'))
	print(dg.edges())
	
	
	


