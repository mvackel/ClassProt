
import numpy as np
import networkx as nx
from itertools import islice, repeat
from joblib import Parallel, delayed
from gephistreamer import streamer, graph as streamGraph

#from math import log
#import matplotlib.pyplot as plt

from typing import List

class Node(object): pass   # Just for typing documentation


class NGraph(nx.Graph):
	_NUM_DSLOTS = 40 + 1  # number of distance slots (0: 0 to <1A, 1: 1 to <2A, ... 39: 39 to <40A, 40: for >=40 Angstrons
	
	def __init__(self, weight=None):
		self.weight = weight
		self._lstJSDs: List[float] = []  # list of Jensen-Shanon Divergence float values, for each k-shortest path
		self._lstNNDs: List[float] = []  # list of Network Node Dispersion float values, for each k-shortest path
		self._lstDistribAvgNodeDists = []  # list of arrays of Average Node Distances Distribution, for each k-shortest path
		self._lstWeightAvgs: List[float] = []  # list of weights averaged by the number of edges
		self._lstJSDHomogs: List[float] = []  # list of Homogeneous Jensen-Shanono Divergences
		self._lstDiameters: List[float] = []  # list of graph diameters, for each k-shortest path value
		self._graphNumbers: List[float] = []  # list of graph numbers, for each k-shortest path value
		self.userData = 0
		super(NGraph, self).__init__()

	
	def JSD(self, k: int = 1) -> float:
		# Returns the Jensen-Shanon divergenge of the graph for the kth shortest path - default: k=1
		# Performance: Try to always call the highest K first to avoid recalculation.
		if (k > len(self._lstJSDs)):
			self.JensenShannonDiverg(k)
		return self._lstJSDs[k - 1]
	
	def NND(self, k: int = 1) -> float:
		# Returns the Network Node Dispersion of the graph for the kth shortest path - default: k=1
		# Performance: Try to always call the highest K first to avoid recalculation.
		if (k > len(self._lstNNDs)):
			self.JensenShannonDiverg(k)  # NNDs are calculated together with JSD
		return self._lstNNDs[k - 1]
	
	def distribAvgNodeDist(self, k: int = 1) -> np.ndarray:
		if k > len(self._lstDistribAvgNodeDists):
			self.JensenShannonDiverg(k)  # distribution of average node distances are calculated together with JSD
		distrib = self._lstDistribAvgNodeDists[k - 1] if k - 1 < len(self._lstDistribAvgNodeDists) else None
		return distrib
	
	def JSDHomog(self, k: int = 1) -> float:
		# Returns the Homogeneous Jensen-Shanon divergenge of the graph for the kth shortest path - default: k=1
		# Performance: Try to always call the highest K first to avoid recalculation.
		if (k > len(self._lstJSDHomogs)):
			self.JensenShannonHomogeneous(k)
		return self._lstJSDHomogs[k - 1]
	
	def diameter(self, k=1):
		if k > len(self._lstDiameters):
			self.JensenShannonDiverg(k)  # diameters are calculated together with JSD
		return self._lstDiameters[k - 1]
	
	# if self.weight is None:
	# 	diam = nx.diameter(self)
	# else:
	# 	diam = nx.diameter(self)
	# # spLambda = lambda G, source: nx.single_source_dijkstra_path_length(G, source, weight=weight)
	# # diam = nx.diameter(self, e=nx.eccentricity(self, sp=spLambda))
	# return diam
	'''
		# G is some graph
		e = nx.eccentricity(G, sp=nx.single_source_dijkstra_path_length)
		d = nx.diameter(G, e)

		# or if you need to specify which edge attribute is the weight
		spl = lambda G, source : nx.single_source_dijkstra_path_length(G, source, weight='myweight')
		d = nx.diameter(G, e=nx.eccentricity(G, sp=spl))
	'''
	
	def k_shortest_path(self, source: Node, target: Node, k: int) -> List[Node]:
		return next(islice(nx.shortest_simple_paths(self, source, target, weight=self.weight), k - 1, None))
	
	def k_shortest_path_length(self, source: Node, target: Node, k: int) -> int:
		# return len(next(islice(nx.shortest_simple_paths(self, source, target, weight=weight), k - 1, None)))
		return len(self.k_shortest_path(source, target, k)) - 1
	
	def single_source_k_path(self, source: Node, k: int) -> dict:
		return {target: self.k_shortest_path(source, target, k) for target in self if target != source}
	
	def single_source_k_path_length(self, source: Node, k: int) -> dict:
		# return { target: self.k_shortest_path_length(source, target, k, weight) for target in self if target != source}
		return {target: self.k_shortest_path_length(source, target, k) for target in self if target != source}
	
	def toK_shortest_paths(self, source: Node, target: Node, k: int) -> List[List[Node]]:
		return list(islice(nx.shortest_simple_paths(self, source, target, weight=self.weight), k))
	
	def toK_shortest_paths_length(self, source: Node, target: Node, k: int) -> List[int]:
		gen = nx.shortest_simple_paths(self, source, target, weight=self.weight)
		if self.weight is None:
			length_func = len
		else:
			def length_func(path):
				return sum(self.adj[u][v][self.weight] for (u, v) in zip(path, path[1:]))
		
		return list(length_func(next(gen)) - 1 for _ in range(k))
	
	def single_source_toK_paths(self, source: Node, k: int) -> dict:
		return {target: self.toK_shortest_paths(source, target, k) for target in self if target != source}
	
	def single_source_toK_paths_length(self, source: Node, k: int) -> dict:
		return {target: self.toK_shortest_paths_length(source, target, k) for target in self if
				target != source}
	
	def buildNodeDistMatricesToK(self, k: int = 2) -> (List[np.array], List[float], List[float]):
		# TODO: rewrite for better performance.
		"""
		@param k: number of k-shortest paths.
				  Ex.: k=2 will return 2 matrices concatenated vertically (by rows)
				       each matrix contains the Node Distance distributions for the k-shortest path.
		@return: a list of arrays containing the k matrices.
				 Each matrix contains the Node Distance distributions for the k-shortest path.
		"""
		
		def buildToKLines(sourceNode: Node, k: int) -> (np.array, np.array, np.array):
			# returns: aNodeDists: array of k lines. Each line contains the distance distribution of each k
			#          aLineWeightAvg: array of k floats: one for each k of value of average of total edges weights
			#          aMaxWeights:    array of k floats: one for each e of the maximum weight (for the diameter)
			dSsp = self.single_source_toK_paths_length(sourceNode, k)
			numDSlots = NGraph._NUM_DSLOTS
			aNodeDists = np.zeros((k, numDSlots))
			aLineWeightAvg = np.zeros(k)  # one column for each line average
			aMaxWeights = np.zeros(k)  # max weights
			for targetNode, lstDists in dSsp.items():
				if sourceNode != targetNode:
					for nK in range(k):
						distK = lstDists[nK] if nK < len(lstDists) else lstDists[-1]
						idx = int(distK) if int(distK) < numDSlots else numDSlots - 1
						aNodeDists[nK, idx] += 1
						aLineWeightAvg[nK] += distK
						if distK > aMaxWeights[nK]:
							aMaxWeights[nK] = distK
			aLineWeightAvg /= len(dSsp.keys())  # dSsp has N-1 keys
			return aNodeDists, aLineWeightAvg, aMaxWeights
		
		numNodes = self.number_of_nodes()
		# buildToKLines is 30 times slower than buildNodeDistMatrixByLine() that uses
		# np.single_source_dijkstra_path_length(), but necessary for k>=2
		lstaNodeDistsMatrices = []
		# construct the list of empty matrices:
		for nm in range(k):
			lstaNodeDistsMatrices.append(np.empty((numNodes, NGraph._NUM_DSLOTS)))
		# populate the empty matrices:
		lstTotalWeightAvg = list(repeat(0.0, k))
		lstDiameters = list(repeat(0.0, k))
		for nNode, node in enumerate(self):
			# aNodeDists = np.zeros((1, MemGraph._NUM_DSLOTS))  # de 0 a 1 Angstrons em [0], etc
			aNodeDists, aLineWeightAvg, aMaxWeights = buildToKLines(node, k)
			for nK in range(k):
				lstaNodeDistsMatrices[nK][nNode] = aNodeDists[nK]
				valAvg = aLineWeightAvg[nK]
				lstTotalWeightAvg[nK] = (lstTotalWeightAvg[nK] + valAvg) / 2.0 if nNode > 0 else valAvg
				# Unweighted graph: Diameter of the graph is the bigger shortest path weight:
				# Weighted graph: get from aMaxWeights:
				maxDist = np.argmax(aNodeDists[nK]) if self.weight is None else aMaxWeights[nK]
				if lstDiameters[nK] < maxDist:
					lstDiameters[nK] = maxDist
		
		return lstaNodeDistsMatrices, lstTotalWeightAvg, lstDiameters
	
	def parBuildToKLines(self, nOrder: int, sourceNode: Node, k: int) -> (int, np.array, np.array, np.array):
		# USED BY parBuildNodeDistMatricesToK()
		dSsp = self.single_source_toK_paths_length(sourceNode, k)
		numDSlots = NGraph._NUM_DSLOTS
		aNodeDists = np.zeros((k, numDSlots))
		aLineWeightAvg = np.zeros(k)  # one column for each line average
		aMaxWeights = np.zeros(k)  # max weights
		for targetNode, lstDists in dSsp.items():
			if sourceNode != targetNode:
				for nK in range(k):
					distK = lstDists[nK] if nK < len(lstDists) else lstDists[-1]
					idx = int(distK) if int(distK) < numDSlots else numDSlots - 1
					aNodeDists[nK, idx] += 1
					aLineWeightAvg[nK] += distK
					if distK > aMaxWeights[nK]:
						aMaxWeights[nK] = distK
		aLineWeightAvg /= len(dSsp.keys())  # dSsp has N-1 keys
		return nOrder, sourceNode, aNodeDists, aLineWeightAvg, aMaxWeights
	
	def parBuildNodeDistMatricesToK(self, k: int = 2) -> (List[np.array], List[float], List[float]):
		# Parallel implementation
		# TODO: rewrite for better performance.
		"""
		@param k: number of k-shortest paths.
				  Ex.: k=2 will return 2 matrices concatenated vertically (by rows)
				       each matrix contains the Node Distance distributions for the k-shortest path.
		@return: a list of arrays containing the k matrices.
				 Each matrix contains the Node Distance distributions for the k-shortest path.
		"""
		
		numNodes = self.number_of_nodes()
		# buildToKLines is 30 times slower than buildNodeDistMatrixByLine() that uses
		# np.single_source_dijkstra_path_length(), but necessary for k>=2
		
		# construct the list of empty matrices:
		# careful: do not use  [ np.zeros(..) ] * k      : this will copy only references to the same array.
		lstaNodeDistsMatrices = [np.zeros((numNodes, NGraph._NUM_DSLOTS)) for i in range(k)]
		lstTotalWeightAvg = [0.0 for i in range(k)]
		lstDiameters = [0.0 for i in range(k)]
		# lstDiameters = list(repeat(0.0, k))   # exampla using repeat
		
		# Parallel procesing:
		lstResultsNodeDists = Parallel(n_jobs=-1, backend='multiprocessing', batch_size='auto') \
			(delayed(self.parBuildToKLines)(nOrder, node, k) for nOrder, node in enumerate(self))
		
		# Sort (unecessary - just to easy debugging....
		lstResultsNodeDists.sort(key=lambda item: item[0])
		
		for result in lstResultsNodeDists:
			# result: nOrder, sourceNode, aNodeDists, aLineWeightAvg, aMaxWeights
			#          [0]      [1]          [2]          [3]           [4]
			nOrder = result[0]
			# sourceNode = result[1]
			aNodeDists = result[2]
			aLineWeightAvg = result[3]
			aMaxWeights = result[4]
			for nK in range(k):
				lstaNodeDistsMatrices[nK][nOrder] = aNodeDists[nK]
				valAvg = aLineWeightAvg[nK]
				lstTotalWeightAvg[nK] = (lstTotalWeightAvg[nK] + valAvg) / 2.0 if nOrder > 0 else valAvg
				maxDist = np.argmax(aNodeDists[nK]) if self.weight is None else aMaxWeights[nK]
				# Unweighted graph: Diameter of the graph is the bigger shortest path weight:
				# Weighted graph: get from aMaxWeights:
				if lstDiameters[nK] < maxDist:
					lstDiameters[nK] = maxDist
		return lstaNodeDistsMatrices, lstTotalWeightAvg, lstDiameters
	
	def buildNodeDistMatrixByLine(self) -> np.array:
		
		def buildLine(sourceNode: Node) -> np.array:
			ssp = nx.single_source_dijkstra_path_length(self, sourceNode, weight=self.weight)
			numDSlots = NGraph._NUM_DSLOTS
			aNodeDists = np.zeros((1, numDSlots))
			for node in ssp:
				if sourceNode != node:
					dist = ssp[node]  # self[sourceNode][node]['Dist']   # dist between sourceNode and node
					idx = int(dist) if int(dist) < numDSlots else numDSlots - 1
					# aNodes = ssp[1][node] # list of nodes in the shortest path between source and node
					aNodeDists[0, idx] += 1
			return aNodeDists
		
		# aNodeDistsMatrix = np.empty((1, MemGraph._NUM_DSLOTS))
		aNodeDistsMatrix = np.zeros((self.number_of_nodes(), NGraph._NUM_DSLOTS))
		for n, node in enumerate(self):
			aNodeDists = np.zeros((1, NGraph._NUM_DSLOTS))  # de 0 a 1 Angstrons em [0], etc
			aNodeDists = buildLine(node)
			# aNodeDists.reshape(1, MemGraph._NUM_DSLOTS)
			aNodeDistsMatrix[n] = aNodeDists
		return aNodeDistsMatrix
	
	def buildNodeDistMatrix(self) -> np.array:
		ssp = list(nx.all_pairs_dijkstra_path_length(self, weight=self.weight))
		numDSlots = NGraph._NUM_DSLOTS
		aNodeDistsMatrix = np.zeros((len(ssp), numDSlots))
		for nnode, nodeData in enumerate(ssp):
			for dist in nodeData[1].values():
				if dist > 0:
					idx = int(dist) if int(dist) < numDSlots else numDSlots - 1
					aNodeDistsMatrix[nnode][idx] += 1
		return aNodeDistsMatrix
	
	def parBuildNLines(self, numBatch: int, lstSourceNodes: List[Node]) -> (int, np.array):
		# numBatch is used to coordinate the parallel processing
		numDSlots = NGraph._NUM_DSLOTS
		aNodeDists = np.zeros((len(lstSourceNodes), numDSlots))
		for nnode, sNode in enumerate(lstSourceNodes):
			ssp = nx.single_source_dijkstra_path_length(self, sNode, weight=self.weight)
			for dist in ssp.values():
				if dist > 0:
					idx = int(dist) if int(dist) < numDSlots else numDSlots - 1
					aNodeDists[nnode, idx] += 1
		return (numBatch, aNodeDists)
	
	def parBuildNodeDistMatrix(self) -> np.array:  # Parallel Implementation
		
		numNodes = self.number_of_nodes()
		aNodeDistsMatrix = np.zeros((numNodes, NGraph._NUM_DSLOTS))
		if numNodes > 10:
			npar = 50
			# for i in range(0, numNodes, npar):
			# 	print(list(self.nodes())[i:i+npar])
			lstAllNodes = list(self.nodes())
			aLstNodes = []
			for i in range(0, numNodes, npar):
				aLstNodes.append(lstAllNodes[i:i + npar])
			# print(aLstNodes)
			
			# lstaNodeDists = Parallel(n_jobs=10, backend="threading")(delayed(self.buildNLines)(lstNodes) for lstNodes in aLstNodes)
			# ltaNodeDists = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(buildNLines)(self, numBatch, lstNodes)
			# 															   for numBatch, lstNodes in enumerate(aLstNodes))
			ltaNodeDists = Parallel(n_jobs=-1, backend='multiprocessing')(
				delayed(self.parBuildNLines)(numBatch, lstNodes)
				for numBatch, lstNodes in enumerate(aLstNodes))
			# lstaNodeDists = Parallel(n_jobs=2, backend='multiprocessing', batch_size=100)\
			#	(delayed(self.buildLine)(node) for node in lstAllNodes)
			for aNodeDists in ltaNodeDists:
				idx = npar * aNodeDists[0]  # [0]: numBatch
				aNodeDistsMatrix[idx:idx + npar] = aNodeDists[1]
		
		return aNodeDistsMatrix
	
	# Jensen Shannon Divergence
	# Returns: (JSDiverg, NDD)
	def JensenShannonDiverg(self, k: int = 2) -> (List[float], List[float]):
		if k > len(self._lstJSDs):
			# TODO: Verify: / (nNodes - 1) or (nNodes):
			numDSlots = NGraph._NUM_DSLOTS
			nNodes = self.number_of_nodes()
			if nNodes > 30:
				# Parallel calc:
				matrices, weightAvgs, diameters = self.parBuildNodeDistMatricesToK(k)
			else:
				# Serial calc:
				matrices, weightAvgs, diameters = self.buildNodeDistMatricesToK(k)
			self._lstWeightAvgs = weightAvgs
			self._lstDiameters = diameters
			lstJSDs = list(repeat(0.0, k))
			lstNNDs = list(repeat(0.0, k))
			self._lstDistribAvgNodeDists = list(repeat(0.0, k))
			for nK in range(k):
				matrices[nK] /= (nNodes - 1)
				# the average Node distributions will be used by
				# JensenShannon2Graphs() and other things:
				self._lstDistribAvgNodeDists[nK] = np.average(matrices[nK], axis=0).reshape(1, numDSlots)
				(nrows, ncols) = matrices[nK].shape
				means = np.average(matrices[nK], 0)
				sumJSD = 0.0
				for nc in range(ncols):
					col = matrices[nK][:, nc]
					sumJSD = sumJSD + sum(
						_p * np.math.log(_p / means[nc]) for _p in col if _p != 0.0 and means[nc] != 0)
				lstJSDs[nK] = np.round(sumJSD / nNodes, 14)  # Jensen-Shanon Divergence.
				lstNNDs[nK] = np.round(lstJSDs[nK] / np.math.log(self.diameter(k=nK) + 1), 14)
			self._lstJSDs = lstJSDs
			self._lstNNDs = lstNNDs
		
		return (self._lstJSDs[:k], self._lstNNDs[:k])
	
	def JensenShannonHomogeneous(self, k: int = 2) -> List[float]:
		# TODO: Slots distribution according to the weights distribution
		if k > len(self._lstJSDHomogs):
			numDSlots = NGraph._NUM_DSLOTS
			# for weighted graphs, distribute in the weight interval (0 .. diam]
			# for unweighted graphs, distribute in the interval [1 .. diam]
			lowerBound = 1 if self.weight is None else 0
			# Should calculate in reverse order to improve performance:
			for nK in range(k, 0, -1):
				diam = self.diameter(nK)
				upperBound = int(diam) if int(diam) < numDSlots else numDSlots - 1
				ugK = self.distribAvgNodeDist(nK)  # Already divided... / (self.number_of_nodes() - 1)
				# valDistrib = self._lstWeightAvgs[nK-1] / (self.number_of_nodes()-1)#diam    # average shortest-path lenght / weighted diameter
				numCols = (upperBound - lowerBound + 1)
				valDistrib = 1.0 / (upperBound - lowerBound + 1)
				homogDistrib = np.zeros((1, numDSlots))
				homogDistrib[0, lowerBound:upperBound + 1] = np.full((1, numCols), valDistrib)
				ugH = homogDistrib  # ==== / (self.number_of_nodes() - 1)
				distrMatrix = np.concatenate((ugK, ugH), axis=0)
				u_avg = np.average(distrMatrix, axis=0)
				(nrows, ncols) = distrMatrix.shape
				JSDu = 0.0
				for nc in range(ncols):
					col = distrMatrix[:, nc]
					JSDu = JSDu + sum(_u * np.math.log(_u / u_avg[nc]) for _u in col if _u != 0)
				JSDu /= 2.0
				self._lstJSDHomogs.append(JSDu)
			self._lstJSDHomogs.reverse()  # recover correct order
		
		return self._lstJSDHomogs[:k]
	
	def JensenShannon2Graphs(self, other: 'NGraph', k: int = 2) -> float:
		# TODO: Verify: / (nNodes - 1) or (nNodes):
		ug1 = self.distribAvgNodeDist(k) / (self.number_of_nodes() - 1)
		ug2 = other.distribAvgNodeDist(k) / (other.number_of_nodes() - 1)
		distrMatrix = np.concatenate((ug1, ug2), axis=0)
		u_avg = np.average(distrMatrix, axis=0)
		(nrows, ncols) = distrMatrix.shape
		JSDu = 0.0
		for nc in range(ncols):
			col = distrMatrix[:, nc]
			JSDu = JSDu + sum(_u * np.math.log(_u / u_avg[nc]) for _u in col if _u != 0)
		JSDu /= 2.0
		return JSDu
	
	def RavettiDissimilarity(self, other: 'NGraph') -> float:
		# TODO: implement the third term of the equation, using Alpha-centrality
		ljsd1, lnnd1 = self.JensenShannonDiverg(k=1)
		ljsd2, lnnd2 = other.JensenShannonDiverg(k=1)
		nnd1 = lnnd1[0]
		nnd2 = lnnd2[0]
		jsdG1G2 = self.JensenShannon2Graphs(other, k=1)
		D_G1G2 = 0.5 * np.math.sqrt(jsdG1G2 / np.math.log(2)) + \
				 0.5 * np.math.fabs(np.math.sqrt(nnd1) - np.math.sqrt(nnd2))
		return D_G1G2
	
	def RavettiDissimilarityModifiedV1(self, other: 'NGraph') -> float:
		# Does not solve Dodec X Desargues
		_, lnnd1 = self.JensenShannonDiverg(k=2)
		_, lnnd2 = other.JensenShannonDiverg(k=2)
		nnd1_k1 = lnnd1[0]
		nnd2_k1 = lnnd2[0]
		nnd1_k2 = lnnd1[1]
		nnd2_k2 = lnnd2[1]
		jsdG1G2 = self.JensenShannon2Graphs(other)
		D_G1G2 = 1 / 3 * np.math.sqrt(jsdG1G2 / np.math.log(2)) + \
				 1 / 3 * np.math.fabs(np.math.sqrt(nnd1_k1) - np.math.sqrt(nnd2_k1)) + \
				 1 / 3 * np.math.fabs(np.math.sqrt(nnd1_k2) - np.math.sqrt(nnd2_k2))
		return D_G1G2
	
	def RavettiDissimilarityModified(self, other: 'NGraph', k=2) -> float:
		_, lNND_G1 = self.JensenShannonDiverg(k)
		_, lNND_G2 = other.JensenShannonDiverg(k)
		
		lWa = list(repeat(1.0 / (2 * k), k))  # normalization factors
		lWb = list(repeat(1.0 / (2 * k), k))
		
		# this can be calculated in direct order because all terms where calculated in the previous lines
		DM_G1G2 = 0
		for nK in range(1, k + 1):
			jsdG1G2_k = self.JensenShannon2Graphs(other, nK)
			nndG1_k = self.NND(nK)
			nndG2_k = other.NND(nK)
			DM_G1G2 = DM_G1G2 + lWa[nK - 1] * np.math.sqrt(jsdG1G2_k / np.math.log(2)) + \
					  lWb[nK - 1] * np.math.fabs(np.math.sqrt(nndG1_k) - np.math.sqrt(nndG2_k))
		
		return DM_G1G2
	
	def graphNumber(self, k: int = 2) -> np.array:
		if k > len(self._graphNumbers):
			self.JensenShannonDiverg(k)
			self.JensenShannonHomogeneous(k)
			
			# lWak = list(repeat(1.0 / (2 * k), k))  # normalization factors
			# lWbk = list(repeat(1.0 / (2 * k), k))
			# Better if Wa + Wb = 1 for any k:
			wa = 0.5
			wb = 0.5
			
			# this can be calculated in direct order because all terms where calculated in the previous lines
			aGN = np.zeros(k)
			for nK in range(1, k + 1):
				jsdHomog_k = self.JSDHomog(nK)
				nnd_k = self.NND(nK)
				# aGN[nK-1] = aGN[nK-1] + lWak[nK-1] * np.math.sqrt( jsdHomog_k / np.math.log(2)) + \
				#		                lWbk[nK-1] * np.math.sqrt(nnd_k)
				aGN[nK - 1] = aGN[nK - 1] + wa * np.math.sqrt(jsdHomog_k / np.math.log(2)) + \
							  wb * np.math.sqrt(nnd_k)
			self._graphNumbers = aGN
		
		return self._graphNumbers
	
	def streamToGephi(self, hostname="localhost", port=8008, workspace="workspace1"):
		gephiWs = streamer.GephiWS(hostname=hostname, port=port, workspace=workspace)
		# stream = streamer.Streamer(streamer.GephiWS(hostname="localhost", port=8090, workspace="workspace1"))
		stream = streamer.Streamer(gephiWs)
		
		# Nodos:
		ln = []
		for nodo in self.nodes():
			stream.add_node(streamGraph.Node(nodo, custom_property=1))
			node_s = streamGraph.Node(nodo, custom_property=1)
			ln.append(node_s)
		
		le = []
		for edge in self.edges(data=True):
			n1 = edge[0]
			n2 = edge[1]
			w = edge[2]['Dist']
			stream.add_edge(streamGraph.Edge(n1, n2, directed=False, weight=w))
			le.append(streamGraph.Edge(n1, n2, directed=False, weight=w))
# ln = [ sGraph.Node(x, nome=) for x, p in gProt.nodes()]


# ================================================================

if __name__ == "__main__":
	pass
