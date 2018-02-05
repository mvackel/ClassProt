from gephistreamer import graph as sGraph
from gephistreamer import streamer

import py2neo as pn

from ClassProtGraph import *


if __name__ == "__main__":
	gephiWs = streamer.GephiWS(hostname="localhost", port=9009, workspace="workspace1")
	# stream = streamer.Streamer(streamer.GephiWS(hostname="localhost", port=8090, workspace="workspace1"))
	stream = streamer.Streamer(gephiWs)
	
	pn.authenticate("localhost:7474", "neo4j", "pxyon123")
	
	dbGraph = pn.Graph()
	
	startTime = time.time()
	
	
	gProt = ClassProtGraph()
	gProt.fromGraphDB(dbGraph, '2JKU')
	
	# Nodos:
	ln = []
	for nodo in gProt.nodes():
		stream.add_node(sGraph.Node(nodo, custom_property=1))
		node_s = sGraph.Node(nodo, custom_property=1)
		ln.append(node_s)
	
	le = []
	for edge in gProt.edges(data=True):
		n1 = edge[0]
		n2 = edge[1]
		w  = edge[2]['Dist']
		stream.add_edge(sGraph.Edge(n1,n2,directed=False, weight=w))
		le.append( sGraph.Edge(n1, n2, directed=False, weight=w) )
	#ln = [ sGraph.Node(x, nome=) for x, p in gProt.nodes()]
	