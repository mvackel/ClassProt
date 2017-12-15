'''
	Ditribution Tools
'''
import math
import numpy as np
from numpy import ndarray


def JensenShannonDivergFromMatrix(matrix:ndarray) -> float :
	# distributions are in columns
	(nrows, ncols) = matrix.shape
	means = np.average(matrix, 0)
	JSD = 0.0
	for nc in range(ncols):
		col = matrix[:,nc]
		JSD = JSD + sum( _p * math.log(_p/means[nc]) for _p in col if _p != 0)
	return JSD
	
def JensenShannonDiverg( P:ndarray, Q:ndarray ) -> float :
	pass
	return 0.0



from Tools import *
from MemGraph import *
import py2neo as pn
from py2neo import *

import networkx as nx
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
	mtst = np.array([[4,5,6,7,8 ,9 ,11, 0],
					 [1,2,3,4,5 ,6 ,7 , 0],
					 [7,6,5,4,3 ,2 ,1 , 0],
					 [2,4,6,8,10,12,14, 0],
					 [1,2,1,1, 1, 1, 1, 0]])
	
	# means:        3.0, 3.8, 4.2, 4.8, 5.4, 6.0, 6.8
	# JSD: 37.4512509419
	
	jsd = JensenShannonDivergFromMatrix(mtst)
	print(jsd)
	
	(entry, casFromPdb) = getPDB('5O75')
	
	# calcula a matrix de distancias:
	# pos 21 reservado para >= 20 Angstrons
	arrNumNodos = np.array([])
	for nca, oCa in enumerate(casFromPdb):
		arrNumNodosDists = np.zeros(41).reshape(1,41)  # de 0 a 1 Angstrons em [0], etc
		dists = calcDistances(casFromPdb, nca)
		for n, dist in enumerate(dists):  #[nca:]):  # sempre a partir do CAlpha em diante
			if dist > 0.0:
				idx = int(dist) if int(dist) <= 40 else 40
				arrNumNodosDists[0,idx] += 1
		#print(f'nca: {nca} => {arrNumNodosDists}')
		if nca == 0:
			arrNumNodos = arrNumNodosDists
			print(oCa)
			print(arrNumNodosDists)
		else:
			arrNumNodos = np.concatenate((arrNumNodos, arrNumNodosDists), axis=0)
		
		
	print(f'shape: {arrNumNodos.shape}')
	jsd = JensenShannonDivergFromMatrix(arrNumNodos)
	print(f'JSDivergence: {jsd}')

	# print('======= MemGraph =======')
	# pn.authenticate("localhost:7474", "neo4j", "pxyon123")
	#
	# dbGraph = pn.Graph()
	#
	# startTime = time.time()
	#
	# gProt = MemGraph()
	# gProt.fromGraphDB(dbGraph, '5O75')
