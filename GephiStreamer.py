from gephistreamer import graph as sGraph
from gephistreamer import streamer

import py2neo as pn

from MemGraph import *


if __name__ == "__main__":
	gephiWs = streamer.GephiWS(hostname="localhost", port=8080, workspace="workspace2")
	# stream = streamer.Streamer(streamer.GephiWS(hostname="localhost", port=8090, workspace="workspace1"))
	stream = streamer.Streamer(gephiWs)
	
	pn.authenticate("localhost:7474", "neo4j", "pxyon123")
	
	dbGraph = pn.Graph()
	
	startTime = time.time()
	
	gProt = MemGraph()
	gProt.fromGraphDB(dbGraph, '2JKU')
	
	# Nodos:
	ln = []
	for nodo in gProt.nodes():
		
		node_s = sGraph.Node(nodo, custom_property=1)
		ln.append(node_s)
		
	#ln = [ sGraph.Node(x, nome=) for x, p in gProt.nodes()]
	