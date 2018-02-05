from itertools import islice, repeat

from NGraph import *
import py2neo as pn
from py2neo import *
from numpy import ndarray
import numpy as np
import networkx as nx
from math import log
import matplotlib.pyplot as plt

from typing import List


class ClassProtGraph(NGraph):
	def __init__(self, dbGraph:pn.Graph = None, idPdb:str = None, weight:str='Dist'):
		self.idPdb = ''
		self.ec = ''
		self.taxid = ''
		self.organism = ''
		super(ClassProtGraph, self).__init__(weight)
		if dbGraph != None and idPdb != None:
			self.fromGraphDB(dbGraph, idPdb)
			self.idPdb = idPdb
	
	def fromGraphDB(self, dbGraph: pn.Graph, idPdb: str) -> (bool, 'ClassProtGraph'):
		'''
		Build the Memgraph graph from Neo4j database entry.

		@param dbGraph: py2neo Graph object already instantiated
		@param idPdb: str: PDB symbol ID 4 letters string
		@return: boolean: Tuple (bool, MemGraph) True if found any edges, false if PDB symbol not found (no edges)
		'''
		res = dbGraph.data(f"""
			MATCH (p:PDB {{ IdPDB:'{idPdb}' }})
			RETURN p.IdPDB, p.EC as ec, p.Taxid as taxid, p.Organism as organism
		""")
		if len(res) > 0:
			for prot in res:
				self.idPdb = idPdb
				self.ec = prot['ec']
				self.taxid = prot['taxid']
				self.organism = prot['organism']
			
			# Graph Contruction:
			res = dbGraph.data(f"""
					MATCH (c1:CAlpha {{ IdPDB:'{idPdb}' }} )-[rela:NEAR_10A]-(c2)
					RETURN c1.ResSeq as rs1, c2.ResSeq as rs2, id(rela) as r, rela.Dist as dist
					""")
			# RETURN id(c1) as c1, id(c2) as c2
			for rel in res:
				# self.add_edge(rel['c1'], rel['c2'], Dist=rel['dist'])
				self.add_edge(rel['rs1'], rel['rs2'], Dist=rel['dist'])
			
			self.idPdb = idPdb
		
		return (len(res) > 0, self)
	
	def graphCuttoff(self, cutoffDist: float = 0) -> 'ClassProtGraph':
		# Returns a new MemGraph with filtered edges where Dist <= cutoffDist
		newG = ClassProtGraph()
		newG.idPdb = self.idPdb
		newG.ec = self.ec
		newG.taxid = self.taxid
		newG.organism = self.organism
		if cutoffDist > 0:
			for edge in self.edges(data=True):
				dist = edge[2]['Dist']
				if dist <= cutoffDist:
					newG.add_edge(edge[0], edge[1], Dist=dist)
		return newG
	
# ===========================================================

import time
from joblib import Parallel, delayed

if __name__ == "__main__":
	bTest1Prot = True
	if bTest1Prot:
		pn.authenticate("localhost:7474", "neo4j", "pxyon123")
		dbGraph = pn.Graph()
		
		startTime = time.time()
		# pdbSym1 = '1LS6' #'3QVU' #'1LS6'  # '2JKU'
		pdbSym1 = '2JKU'  # 35  CAs #'1LS6' #'3QVU' #'1LS6'  # '2JKU'
		pdbSym1 = '1HF9'  # 41  CAs
		pdbSym1 = '1GMJ'  # 65  CAs
		pdbSym1 = '3PC7'  # 80  CAs
		pdbSym1 = '4RT5'  # 100 CAs
		# pdbSym1 = '1FM8'  # 212 CAs
		# pdbSym1 = '5KDR'  # 302 CAs
		# pdbSym1 = '3IVT'  # 400 CAs
		# pdbSym1 = '3CUX'  # 501 CAs
		# pdbSym1 = '3BG9'  # 601 CAs  4.1/4.1/3.2
		# pdbSym1 = '3TV5'  # 701 CAs  5.1/5.1/3.5
		# pdbSym1 = '1QCN'  # 803 CAs/7644  7.3/7.3/3.96
		# pdbSym1 = '3TW7'  # 1004 CAs/9387  11.5/11.6/5.35
		# pdbSym1 = '3ACZ'  # 1540 CAs/16187  32.1/33.0/10.87
		# pdbSym1 = '5CSL'  # 2097 CAs/18348  50.5/51.1/15.74
		
		print(f'==== {pdbSym1}')
		gProt1 = ClassProtGraph()
		ok = gProt1.fromGraphDB(dbGraph, pdbSym1)
		
		if ok:
			print(f'---- Duration: {time.time()-startTime}')
			print(f'---- {pdbSym1}: Number of Edges: {gProt1.size()}')
			
			numTestes = 1
			startTime = time.time()
			nodeDists = 0
			for i in range(numTestes):
				nodeDists = gProt1.buildNodeDistMatrixByLine()
			print(f'---- Duration (1): {time.time()-startTime}')
			# print(nodeDists[0])
			print(nodeDists[1])
			
			startTime = time.time()
			for i in range(numTestes):
				nodeDists = gProt1.buildNodeDistMatrix()
			print(f'---- Duration (2): {time.time()-startTime}')
			# print(nodeDists[0])
			print(nodeDists[1])
			
			startTime = time.time()
			for i in range(numTestes):
				nodeDists = gProt1.parBuildNodeDistMatrix()
			print(f'---- Duration(3): {time.time()-startTime}')
			# print(nodeDists[0])
			print(nodeDists[1])
	
	bTest2Prots = False
	if bTest2Prots:
		pn.authenticate("localhost:7474", "neo4j", "pxyon123")
		dbGraph = pn.Graph()
		
		startTime = time.time()
		
		pdbSym1 = '1LS6'  # '3QVU' #'1LS6'  # '2JKU'
		pdbSym2 = '2D06'  # '1Z28'   #'1IN1'
		
		print(f'==== {pdbSym1}')
		gProt1 = ClassProtGraph()
		ok = gProt1.fromGraphDB(dbGraph, pdbSym1)
		
		if ok:
			# gProtEdges = gProt.edges(data=True)
			# g5Edges = [ (x[0],x[1],x[2]) for x in gProtEdges if x[2]['Dist'] <= 5 ]
			# gProt5 = gProt.edge_subgraph()
			# gProt5 = gProt.copy()
			# for edge in gProt.edges():
			#	if edge[2]['Dist'] > 5:
			#		gProt5.remove_edge(*edge)
			gProt5 = nx.Graph()
			# edgs = gProt.edges(data=True)
			for edge in gProt1.edges(data=True):
				dist = edge[2]['Dist']
				if dist <= 5:
					gProt5.add_edge(edge[0], edge[1], Dist=dist)
			
			print(f'---- Duration: {time.time()-startTime}')
			print(f'---- {pdbSym1}: Number of Edges: {gProt1.size()}')
			print(f'---- Prot5: Number of Edges: {gProt5.size()}')
			
			# nodeDist = gProt1.buildLine(659)
			# print(f'nodeDist: {nodeDist}')
			
			nodeDists = gProt1.buildNodeDistMatrix()
			# print(f'{nodeDists}')
			
			jsd, nnd = gProt1.JensenShannonDiverg()
			print(f'-- Jensen-Shannon Divergence: {jsd}')
			print(f'-- Network Node Dispersion NND: {nnd}')
		
		# =======  Segunda proteina:
		startTime = time.time()
		gProt2 = ClassProtGraph()
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
	# print(f'---- Similarity[8, 7, 6, 5] = {gProt.calcSimilarityVector()}')
	
	bMatrixBuildTest = False
	if bMatrixBuildTest:
		pn.authenticate("localhost:7474", "neo4j", "pxyon123")
		dbGraph = pn.Graph()
		
		lstSym = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R',
				  '4GRA', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA']
		lstSym = ['2D06', '3QVU', '3QVV', '3U3J', '3U3K']
		
		print(f'Len: {len(lstSym)}')
		start = time.time()
		lstSerialProt = [ClassProtGraph(dbGraph, sym) for sym in lstSym]
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
		lstSym = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R',
				  '4GRA', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA']
		
		print(f'Len: {len(lstSym)}')
		start = time.time()
		lstSerialProt = [ClassProtGraph(dbGraph, sym) for sym in lstSym]
		print(f'duration (serial): {time.time() - start}')
		
		start = time.time()
		## PIOR TEMPO EM PARALELO..... ?
		## NAO FUNCIONA COM MULTIPROCESS (dbGraph nao serializavel)
		lstProts = Parallel(n_jobs=-1, backend="threading")(
			# delayed( MemGraph().fromGraphDB )(dbGraph, sym) for sym in lstSym )
			delayed(ClassProtGraph)(dbGraph, sym) for sym in lstSym)
		print(f'duration (parallel): {time.time() - start}')
		print(lstProts)
	
	plot = False
	if plot:
		plt.figure(1)
		nx.draw_networkx(gProt1, with_labels=True)
		plt.figure(2)
		nx.draw_networkx(gProt5, with_labels=True)
		
		# gex = nx.Graph()
		# gex.add_nodes_from(range(100,110))
		# nx.draw(gex)
		plt.show()
