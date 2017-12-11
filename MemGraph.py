import py2neo as pn
from py2neo import *

import networkx as nx
import matplotlib.pyplot as plt

from typing import List

class MemGraph(nx.Graph):
	def fromGraphDB(self, dbGraph:pn.Graph, idPdb:str) -> bool:
		res = dbGraph.data(f"""
				MATCH (c1:CAlpha {{ IdPDB:'{idPdb}' }} )-[rela:NEAR_10A]-(c2)
				RETURN id(c1) as c1, id(rela) as r, id(c2) as c2, rela.Dist as dist
				""")
		for rel in res:
			self.add_edge(rel['c1'], rel['c2'], Dist=rel['dist'])
		
		return len(res) > 0
	
	def graphCuttoff(self, cutoffDist:float=0) -> 'MemGraph':
		newG = MemGraph()
		if cutoffDist > 0:
			for edge in self.edges(data=True):
				dist = edge[2]['Dist']
				if dist <= cutoffDist:
					newG.add_edge(edge[0], edge[1], Dist=dist)
		return newG
		
	
	def calcSimilarity(self) -> float:
		#TODO: IMPLEMENTAR
		return self.size()/1000
	
	def calcSimilarityCuttoff(self, cutoffDist:float=0) -> float:
		if cutoffDist == 0:
			sim = self.calcSimilarity()
		else:
			mg = self.graphCuttoff(cutoffDist)
			sim = mg.calcSimilarity()
		return sim
	
	def calcSimilarityVector(self) -> List[float]:
		aSim = [0,0,0,0]
		for n,cutoff in enumerate( range(8, 5-1, -1) ):
			aSim[n] = self.calcSimilarityCuttoff(cutoff)
		return aSim
	
	
import time

if __name__ == "__main__":
	pn.authenticate("localhost:7474","neo4j","pxyon123")
	
	dbGraph = pn.Graph()
	
	startTime = time.time()
	
	gProt = MemGraph()
	gProt.fromGraphDB( dbGraph, '5O75')
	
	#gProtEdges = gProt.edges(data=True)
	#g5Edges = [ (x[0],x[1],x[2]) for x in gProtEdges if x[2]['Dist'] <= 5 ]
	#gProt5 = gProt.edge_subgraph()
	#gProt5 = gProt.copy()
	#for edge in gProt.edges():
	#	if edge[2]['Dist'] > 5:
	#		gProt5.remove_edge(*edge)
	gProt5 = nx.Graph()
	#edgs = gProt.edges(data=True)
	for edge in gProt.edges(data=True):
		dist = edge[2]['Dist']
		if dist <=5:
			gProt5.add_edge( edge[0], edge[1], Dist=dist )
		
	
	print(f'---- Duration: {time.time()-startTime}')
	print(f'---- Prot : Number of Edges: {gProt.size()}')
	print(f'---- Prot5: Number of Edges: {gProt5.size()}')

	print(f'---- Similarity[8, 7, 6, 5] = {gProt.calcSimilarityVector()}')

	plt.figure(1)
	nx.draw_networkx(gProt, with_labels=True)
	plt.figure(2)
	nx.draw_networkx(gProt5, with_labels=True)

	#gex = nx.Graph()
	#gex.add_nodes_from(range(100,110))
	#nx.draw(gex)
	plt.show()