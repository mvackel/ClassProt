import py2neo as pn
from py2neo import *
from numpy import ndarray
import numpy as np
import networkx as nx
from math import log
import matplotlib.pyplot as plt

from typing import List

class MemGraph(nx.Graph):
	def fromGraphDB(self, dbGraph:pn.Graph, idPdb:str) -> bool:
		res = dbGraph.data(f"""
				MATCH (c1:CAlpha {{ IdPDB:'{idPdb}' }} )-[rela:NEAR_10A]-(c2)
				RETURN c1.ResSeq as rs1, c2.ResSeq as rs2, id(rela) as r, rela.Dist as dist
				""")
		# RETURN id(c1) as c1, id(c2) as c2
		for rel in res:
			#self.add_edge(rel['c1'], rel['c2'], Dist=rel['dist'])
			self.add_edge(rel['rs1'], rel['rs2'], Dist=rel['dist'])

		
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
	
	def buildLine(self, sourceNode , numCategories=40) -> np.array:
		ssp = nx.single_source_dijkstra_path_length(self, sourceNode, weight='Dist' )
		aNodeDists = np.zeros(numCategories+1).reshape(1,numCategories+1)
		#visited:List = []
		for node in ssp:
			if sourceNode != node:
				dist = ssp[node]   #self[sourceNode][node]['Dist']   # dist between sourceNode and node
				idx = int(dist) if int(dist) <= numCategories else numCategories
				#aNodes = ssp[1][node] # list of nodes in the shortest path between source and node
				aNodeDists[0, idx] += 1
		return aNodeDists
			
			
	
	def buidNodeDistMatrix(self) -> np.array:
		aNodeDistsMatrix = np.array([])
		for n, node in enumerate(self):
			aNodeDists = np.zeros(41).reshape(1, 41)  # de 0 a 1 Angstrons em [0], etc
			aNodeDists = self.buildLine(node)
			aNodeDists.reshape(1, 41)
			if n == 0:
				aNodeDistsMatrix = aNodeDists
			else:
				aNodeDistsMatrix = np.concatenate((aNodeDistsMatrix, aNodeDists), axis=0)
		return aNodeDistsMatrix
	
	# Jensen Shannon Divergence
	def JensenShannonDiverg(self):
		matrix = self.buidNodeDistMatrix()
		(nrows, ncols) = matrix.shape
		means = np.average(matrix, 0)
		JSD = 0.0
		for nc in range(ncols):
			col = matrix[:, nc]
			JSD = JSD + sum(_p * np.math.log(_p / means[nc]) for _p in col if _p != 0)
		return JSD
	
	# Network Node Dispersion
	def NND(self) -> float:
		diam = nx.diameter(self)
		nnd = self.JensenShannonDiverg() / np.math.log( diam + 1 )
		return nnd


import time

if __name__ == "__main__":
	pn.authenticate("localhost:7474","neo4j","pxyon123")
	
	dbGraph = pn.Graph()
	
	startTime = time.time()
	
	gProt = MemGraph()
	gProt.fromGraphDB( dbGraph, '2JKU')

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


	nodeDist = gProt.buildLine(659)
	print(f'nodeDist: {nodeDist}')
	
	nodeDists = gProt.buidNodeDistMatrix()
	#print(f'{nodeDists}')
	
	jsd = gProt.JensenShannonDiverg()
	print(f'Jensen-Shannon Divergence: {jsd}')
	
	nnd = gProt.NND()
	print(f'Network Node Dispersion NND: {nnd}')
	

	#print(f'---- Similarity[8, 7, 6, 5] = {gProt.calcSimilarityVector()}')

	plot = True
	if plot:
		plt.figure(1)
		nx.draw_networkx(gProt, with_labels=True)
		plt.figure(2)
		nx.draw_networkx(gProt5, with_labels=True)
	
		#gex = nx.Graph()
		#gex.add_nodes_from(range(100,110))
		#nx.draw(gex)
		plt.show()