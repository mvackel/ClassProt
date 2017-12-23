from itertools import islice

import py2neo as pn
from py2neo import *
from numpy import ndarray
import numpy as np
import networkx as nx
from math import log
import matplotlib.pyplot as plt

from typing import List


class MemGraph(nx.Graph):
	_NUM_DSLOTS = 40+1    # number of distance slots (0: 0 to <1A, 1: 1 to <2A, ... 39: 39 to <40A, 40: for >=40 Angstrons
	
	def __init__(self, dbGraph:pn.Graph=None, idPdb:str=None):
		self._JSD = None
		self._NND = None
		self._distribAvgNodeDist = np.zeros((1, 41))
		super(MemGraph, self).__init__()
		if dbGraph != None and idPdb != None:
			self.fromGraphDB(dbGraph, idPdb)
		
	def fromGraphDB(self, dbGraph:pn.Graph, idPdb:str) -> (bool, 'MemGraph'):
		res = dbGraph.data(f"""
				MATCH (c1:CAlpha {{ IdPDB:'{idPdb}' }} )-[rela:NEAR_10A]-(c2)
				RETURN c1.ResSeq as rs1, c2.ResSeq as rs2, id(rela) as r, rela.Dist as dist
				""")
		# RETURN id(c1) as c1, id(c2) as c2
		for rel in res:
			#self.add_edge(rel['c1'], rel['c2'], Dist=rel['dist'])
			self.add_edge(rel['rs1'], rel['rs2'], Dist=rel['dist'])

		
		return (len(res) > 0, self)
	
	def graphCuttoff(self, cutoffDist:float=0) -> 'MemGraph':
		newG = MemGraph()
		if cutoffDist > 0:
			for edge in self.edges(data=True):
				dist = edge[2]['Dist']
				if dist <= cutoffDist:
					newG.add_edge(edge[0], edge[1], Dist=dist)
		return newG
	
	def calcSimilarity(self) -> float:
		# TODO: IMPLEMENTAR
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
	
	from itertools import islice
	def k_shortest_path(self, source, target, k, weight=None):
		return next(islice(nx.shortest_simple_paths(self, source, target, weight=weight), k - 1, None))
	
	def k_shortest_path_length(self, source, target, k, weight=None):
		#return len(next(islice(nx.shortest_simple_paths(self, source, target, weight=weight), k - 1, None)))
		return len( self.k_shortest_path(source, target, k, weight=weight) ) - 1
	
	def single_source_k_path(self, source, k, weight=None ) -> dict:
		return { target: self.k_shortest_path(source, target, k, weight) for target in self if target != source}
		
	def single_source_k_path_length(self, source, k, weight=None ) -> dict:
		#return { target: self.k_shortest_path_length(source, target, k, weight) for target in self if target != source}
		return { target: self.k_shortest_path_length(source, target, k, weight) for target in self if target != source}
	
	def buildKLine(self, sourceNode: Node, k:int=1, weight='Dist') -> np.array:
		ssp = self.single_source_k_path_length(sourceNode, k, weight)
		numDSlots = MemGraph._NUM_DSLOTS
		aNodeDists = np.zeros((1, numDSlots))
		for node in ssp:
			if sourceNode != node:
				dist = ssp[node]  # self[sourceNode][node]['Dist']   # dist between sourceNode and node
				idx = int(dist) if int(dist) < numDSlots else numDSlots - 1
				aNodeDists[0, idx] += 1
		return aNodeDists
	
	def buildNodeDistMatrixKByLine(self, k:int=1) -> np.array:
		# TODO: rewrite for better performance.
		# 30 times slower than buildNodeDistMatrixByLine(), but necessary for k>=2
		aNodeDistsMatrix = np.empty((1, MemGraph._NUM_DSLOTS))
		for n, node in enumerate(self):
			aNodeDists = np.zeros((1, MemGraph._NUM_DSLOTS))  # de 0 a 1 Angstrons em [0], etc
			aNodeDists = self.buildKLine(node, k)
			#aNodeDists.reshape(1, MemGraph._NUM_DSLOTS)
			aNodeDistsMatrix = np.concatenate((aNodeDistsMatrix, aNodeDists), axis=0)
		return aNodeDistsMatrix
	
	def buildLine(self, sourceNode:Node, weight='Dist') -> np.array:
		ssp = nx.single_source_dijkstra_path_length(self, sourceNode, weight)
		numDSlots = MemGraph._NUM_DSLOTS
		aNodeDists = np.zeros((1, numDSlots))
		for node in ssp:
			if sourceNode != node:
				dist = ssp[node]   #self[sourceNode][node]['Dist']   # dist between sourceNode and node
				idx = int(dist) if int(dist) < numDSlots else numDSlots-1
				#aNodes = ssp[1][node] # list of nodes in the shortest path between source and node
				aNodeDists[0, idx] += 1
		return aNodeDists
			
	def buildNodeDistMatrixByLine(self) -> np.array:
		aNodeDistsMatrix = np.empty((1, MemGraph._NUM_DSLOTS))
		for n, node in enumerate(self):
			aNodeDists = np.zeros((1, MemGraph._NUM_DSLOTS))  # de 0 a 1 Angstrons em [0], etc
			aNodeDists = self.buildLine(node)
			#aNodeDists.reshape(1, MemGraph._NUM_DSLOTS)
			aNodeDistsMatrix = np.concatenate((aNodeDistsMatrix, aNodeDists), axis=0)
		return aNodeDistsMatrix

	def buildNodeDistMatrix(self) -> np.array:
		ssp = list(nx.all_pairs_dijkstra_path_length(self, weight='Dist'))
		numDSlots = MemGraph._NUM_DSLOTS
		aNodeDistsMatrix = np.zeros((len(ssp), numDSlots))
		for nnode, nodeData in enumerate(ssp):
			for dist in nodeData[1].values():
				if dist > 0:
					idx = int(dist) if int(dist) < numDSlots else numDSlots - 1
					aNodeDistsMatrix[nnode][idx] += 1
		return aNodeDistsMatrix
	
		
	def buildNLines(self, lstSourceNodes:List[Node]) -> np.array:
		numDSlots = MemGraph._NUM_DSLOTS
		aNodeDists = np.zeros((len(lstSourceNodes), numDSlots))
		for nnode, sNode in enumerate(lstSourceNodes):
			ssp = nx.single_source_dijkstra_path_length(self, sNode, weight='Dist' )
			for dist in ssp.values():
				if dist > 0:
					idx = int(dist) if int(dist) < numDSlots else numDSlots-1
					aNodeDists[nnode, idx] += 1
		return aNodeDists
	
	import itertools as iter
	from itertools import islice
	
	from joblib import Parallel, delayed
	def buildNodeDistMatrix2(self) -> np.array:    # 22222
		numNodes = self.number_of_nodes()
		aNodeDistsMatrix = np.array([])
		if numNodes > 40:
			npar = 150
			# for i in range(0, numNodes, npar):
			# 	print(list(self.nodes())[i:i+npar])
			lstAllNodes = list(self.nodes())
			aLstNodes = []
			for i in range(0,numNodes,npar):
				aLstNodes.append(lstAllNodes[i:i+npar])
			#print(aLstNodes)
				
			#lstaNodeDists = Parallel(n_jobs=10, backend="threading")(delayed(self.buildNLines)(lstNodes) for lstNodes in aLstNodes)
			lstaNodeDists = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(self.buildNLines)(lstNodes) for lstNodes in aLstNodes)
			#lstaNodeDists = Parallel(n_jobs=2, backend='multiprocessing', batch_size=100)\
			#	(delayed(self.buildLine)(node) for node in lstAllNodes)
			
			for n, aNodeDists in enumerate(lstaNodeDists):
				if n == 0:
					aNodeDistsMatrix = aNodeDists
				else:
					aNodeDistsMatrix = np.concatenate((aNodeDistsMatrix, aNodeDists), axis=0)

		return aNodeDistsMatrix

	# Jensen Shannon Divergence
	# Returns: (JSDiverg, NDD)
	def JensenShannonDiverg(self) -> (float, float):
		if self._JSD is None:
			# TODO: Verify: / (nNodes - 1) or (nNodes):
			nNodes = self.number_of_nodes()
			matrix = self.buidNodeDistMatrix() / (nNodes - 1)
			# the average Node distribution will be used by JensenShannon2Graphs():
			self._distribAvgNodeDist = np.average(matrix, axis=0).reshape(1, 41)
			(nrows, ncols) = matrix.shape
			means = np.average(matrix, 0)
			JSD = 0.0
			for nc in range(ncols):
				col = matrix[:, nc]
				JSD = JSD + sum(_p * np.math.log(_p / means[nc]) for _p in col if _p != 0)
			JSD /= nNodes
			diam = nx.diameter(self)
			NND = JSD / np.math.log( diam + 1 )
			self._JSD = JSD
			self._NND = NND
		else:
			JSD = self._JSD
			NND = self._NND

		return (JSD, NND)

	def JensenShannonHomogeneous(self) -> (float, float):
		# if self._JSD_Hom is None:
		# 	# TODO: Verify: / (nNodes - 1) or (nNodes):
		# 	nNodes = self.number_of_nodes()
		# 	numDSlots = MemGraph._NUM_DSLOTS
		# 	matrix = np.ones((nNodes, numDSlots)) * ( / (nNodes - 1)
		# 	# the average Node distribution will be used by JensenShannon2Graphs():
		# 	self._distribAvgNodeDist = np.average(matrix, axis=0).reshape(1, 41)
		# 	(nrows, ncols) = matrix.shape
		# 	means = np.average(matrix, 0)
		# 	JSD = 0.0
		# 	for nc in range(ncols):
		# 		col = matrix[:, nc]
		# 		JSD = JSD + sum(_p * np.math.log(_p / means[nc]) for _p in col if _p != 0)
		# 	JSD /= nNodes
		# 	diam = nx.diameter(self)
		# 	NND = JSD / np.math.log(diam + 1)
		# 	self._JSD = JSD
		# 	self._NND = NND
		# else:
		# 	JSD = self._JSD
		# 	NND = self._NND
		#
		# return (JSD, NND)
		pass
	
	# Network Node Dispersion
	def NND(self) -> float:
		nnd = self._NND
		if self._NND is None:
			jsd, nnd = self.JensenShannonDiverg()
		return nnd
	
	def distribNodeDist(self) -> np.ndarray:
		if not self._distribAvgNodeDist.any():
			self.JensenShannonDiverg()
		return self._distribAvgNodeDist
		
	def JensenShannon2Graphs(self, other):
		# TODO: Verify: / (nNodes - 1) or (nNodes):
		ug1 = self.distribNodeDist() / (self.number_of_nodes() - 1)
		ug2 = other.distribNodeDist() / (other.number_of_nodes() - 1)
		distrMatrix = np.concatenate((ug1, ug2), axis=0)
		u_avg = np.average(distrMatrix, axis=0)
		(nrows, ncols) = distrMatrix.shape
		JSDu = 0.0
		for nc in range(ncols):
			col = distrMatrix[:, nc]
			JSDu = JSDu + sum(_u * np.math.log(_u / u_avg[nc]) for _u in col if _u != 0)
		JSDu /= 2.0
		return JSDu
	
	def RavettiDissimilarity(self, other) -> float:
		# TODO: implement the third term of the equation, using Alpha-centrality
		jsd1, nnd1 = self.JensenShannonDiverg()
		jsd2, nnd2 = other.JensenShannonDiverg()
		jsdG1G2 = self.JensenShannon2Graphs(other)
		D_G1G2 = 0.5 * np.math.sqrt(jsdG1G2 / np.math.log(2)) + \
				 0.5 * np.math.fabs(np.math.sqrt(nnd1) - np.math.sqrt(nnd2))
		return D_G1G2

import time
from joblib import Parallel, delayed

if __name__ == "__main__":
	bTest1Prot = True
	if bTest1Prot:
		pn.authenticate("localhost:7474", "neo4j", "pxyon123")
		dbGraph = pn.Graph()
		
		startTime = time.time()
		pdbSym1 = '1LS6' #'3QVU' #'1LS6'  # '2JKU'
		print(f'==== {pdbSym1}')
		gProt1 = MemGraph()
		ok = gProt1.fromGraphDB(dbGraph, pdbSym1)
		
		if ok:
			print(f'---- Duration: {time.time()-startTime}')
			print(f'---- {pdbSym1}: Number of Edges: {gProt1.size()}')
			
			startTime = time.time()
			for i in range(5):
				nodeDists = gProt1.buildNodeDistMatrixByLine()
			print(f'---- Duration: {time.time()-startTime}')
			#print(nodeDists[0])
			print(nodeDists[1])

			startTime = time.time()
			for i in range(5):
				nodeDists = gProt1.buildNodeDistMatrix()
			print(f'---- Duration: {time.time()-startTime}')
			#print(nodeDists[0])
			print(nodeDists[1])

			startTime = time.time()
			for i in range(5):
				nodeDists = gProt1.buildNodeDistMatrix2()
			print(f'---- Duration: {time.time()-startTime}')
			#print(nodeDists[0])
			print(nodeDists[1])
	
	bTest2Prots = False
	if bTest2Prots:
		pn.authenticate("localhost:7474", "neo4j", "pxyon123")
		dbGraph = pn.Graph()
		
		startTime = time.time()
	
		pdbSym1 = '1LS6' #'3QVU' #'1LS6'  # '2JKU'
		pdbSym2 = '2D06' #'1Z28'   #'1IN1'
	
		print(f'==== {pdbSym1}')
		gProt1 = MemGraph()
		ok = gProt1.fromGraphDB(dbGraph, pdbSym1)
	
		if ok:
			#gProtEdges = gProt.edges(data=True)
			#g5Edges = [ (x[0],x[1],x[2]) for x in gProtEdges if x[2]['Dist'] <= 5 ]
			#gProt5 = gProt.edge_subgraph()
			#gProt5 = gProt.copy()
			#for edge in gProt.edges():
			#	if edge[2]['Dist'] > 5:
			#		gProt5.remove_edge(*edge)
			gProt5 = nx.Graph()
			# edgs = gProt.edges(data=True)
			for edge in gProt1.edges(data=True):
				dist = edge[2]['Dist']
				if dist <=5:
					gProt5.add_edge( edge[0], edge[1], Dist=dist )
	
			print(f'---- Duration: {time.time()-startTime}')
			print(f'---- {pdbSym1}: Number of Edges: {gProt1.size()}')
			print(f'---- Prot5: Number of Edges: {gProt5.size()}')

	
			# nodeDist = gProt1.buildLine(659)
			# print(f'nodeDist: {nodeDist}')
	
			nodeDists = gProt1.buidNodeDistMatrix()
			#print(f'{nodeDists}')
	
			jsd, nnd = gProt1.JensenShannonDiverg()
			print(f'-- Jensen-Shannon Divergence: {jsd}')
			print(f'-- Network Node Dispersion NND: {nnd}')
	
		# =======  Segunda proteina:
		startTime = time.time()
		gProt2 = MemGraph()
		print(f'==== {pdbSym2}')
		ok = gProt2.fromGraphDB(dbGraph, pdbSym2)
	
		print(f'---- Duration: {time.time()-startTime}')
		print(f'---- {pdbSym2}: Number of Edges: {gProt2.size()}')
	
		if ok:
			jsd, nnd = gProt2.JensenShannonDiverg()
			print(f'-- Jensen-Shannon Divergence: {jsd}')
			print(f'-- Network Node Dispersion NND: {nnd}')
			jsdG1G2 = gProt2.JensenShannon2Graphs(gProt1)
			print(f'== Jansen Shannon G1, G2 = {jsdG1G2}')
	
			dissimilarity = gProt2.RavettiDissimilarity(gProt1)
			print(f'===== Ravetti Dissimilarity: D({pdbSym2}, {pdbSym1}) = {dissimilarity}')
		#print(f'---- Similarity[8, 7, 6, 5] = {gProt.calcSimilarityVector()}')


	bMatrixBuildTest = False
	if bMatrixBuildTest:
		pn.authenticate("localhost:7474", "neo4j", "pxyon123")
		dbGraph = pn.Graph()

		lstSym = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA']
		lstSym = ['2D06', '3QVU', '3QVV', '3U3J', '3U3K']

		print(f'Len: {len(lstSym)}')
		start = time.time()
		lstSerialProt = [MemGraph(dbGraph, sym) for sym in lstSym]
		print(f'Mount duration (serial): {time.time() - start}')
		start = time.time()
		for mp in lstSerialProt:
			mp.buidNodeDistMatrix()
		dur = time.time() - start
		print(f'Matrix duration (serial): {dur} - Average: {dur/len(lstSym)} / prot')


	
	bParallelTest = False
	if bParallelTest:
		# Parallel:
		lstSym = ['2D06', '3QVU', '3QVV', '3U3J', '3U3K']
		lstSym = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA']
	
		print(f'Len: {len(lstSym)}')
		start = time.time()
		lstSerialProt = [MemGraph(dbGraph, sym) for sym in lstSym]
		print(f'duration (serial): {time.time() - start}')
	
		start = time.time()
		## PIOR TEMPO EM PARALELO..... ?
		## NAO FUNCIONA COM MULTIPROCESS (dbGraph nao serializavel)
		lstProts = Parallel(n_jobs=-1, backend="threading")(
					#delayed( MemGraph().fromGraphDB )(dbGraph, sym) for sym in lstSym )
					delayed( MemGraph )(dbGraph, sym) for sym in lstSym )
		print(f'duration (parallel): {time.time() - start}')
		print(lstProts)
	


	plot = False
	if plot:
		plt.figure(1)
		nx.draw_networkx(gProt1, with_labels=True)
		plt.figure(2)
		nx.draw_networkx(gProt5, with_labels=True)
	
		#gex = nx.Graph()
		#gex.add_nodes_from(range(100,110))
		#nx.draw(gex)
		plt.show()
		
		
