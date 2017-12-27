from MemGraph import *

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
	
	
	lDodec = [(1, 2), (1, 10), (1, 11), (2, 3), (2, 12), (3, 4), (3, 13), (4, 5), (4, 14),
	          (5, 6), (5, 15), (6, 7), (6, 16), (7, 8), (7, 17), (8, 9), (8, 18), (9, 10), (9, 19),
	          (10, 20), (11, 13), (11, 19), (12, 14), (12, 20), (13, 15), (14, 16), (15, 17), (16, 18),
	          (17, 19), (18, 20)]

	gDodec = MemGraph()
	gDodec.add_edges_from(lDodec)
	
	lDesargues = [(1, 2), (1, 10), (1, 11), (2, 3), (2, 12), (3, 4), (3, 13), (4, 5), (4, 14), (5, 6),
	              (5, 15), (6, 7), (6, 16), (7, 8), (7, 17), (8, 9), (8, 18), (9, 10), (9, 19),
	              (10, 20), (11, 14), (11, 18), (12, 15), (12, 19), (13, 16), (13, 20), (14, 17),
	              (15, 18), (16, 19), (17, 20)]
	
	gDesargues = MemGraph()
	gDesargues.add_edges_from(lDesargues)
	
	# print(gDodec.single_source_k_path(1,2))
	# print(gDesargues.single_source_k_path(1, 2))
	# print(gDodec.single_source_k_path_length(1, 2))
	# print(gDesargues.single_source_k_path_length(1, 2))
	
	bConf = False
	if bConf:
		print(gDodec.buildLine(1,weight=None))
		print(gDodec.buildToKLines(1,k=1,weight=None))
		print(gDesargues.buildLine(1,weight=None))
		print(gDesargues.buildToKLines(1,k=1,weight=None))
		print('--------------')
		print(gDodec.buildToKLines(1,k=2,weight=None))
		print(gDesargues.buildToKLines(1,k=2,weight=None))
		print('=================')
		print(gDodec.buildNodeDistMatricesToK(2, weight=None))
		print('--------------')
		print(gDesargues.buildNodeDistMatricesToK(2, weight=None))
	
	bPerf = False
	if bPerf:
		start = time.time()
		for i in range(100):
			gDodec.buildNodeDistMatricesToK(2, weight=None)
		print(f'Duration: {time.time() - start}')
	
	bGNumber = False
	if bGNumber:
		start = time.time()
		graphNumberDodec = gDodec.graphNumber(weight=None)
		print(f'Graph Number Dodec    : {graphNumberDodec}')

		graphNumberDesargues = gDesargues.graphNumber(weight=None)
		print(f'Graph Number Desargues: {graphNumberDesargues}')
		print(f'Duration: {(time.time() - start)/2}')

	bGNumberPdb = True
	if bGNumberPdb:
		pn.authenticate("localhost:7474", "neo4j", "pxyon123")
		dbGraph = pn.Graph()

		#lstSym = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA']
		#lstSym = ['2D06', '3QVU', '3QVV', '3U3J', '3U3K']
		lstSym = ['5O75', '3U3K']
		lstSym = ['3U3K']

		print(f'Len: {len(lstSym)}')
		start = time.time()
		lstSerialProt = [MemGraph(dbGraph, sym) for sym in lstSym]
		lstSerialProt = [_graph.graphCuttoff(6.0) for _graph in lstSerialProt]
		print(f'Mount duration (serial): {time.time() - start}')
		start = time.time()
		for mp in lstSerialProt:
			print(f'IdPdb: {mp.idPdb}   graphNumber = {mp.graphNumber()}')
		dur = time.time() - start
		print(f'Matrix duration (serial): {dur} -- Average: {dur/len(lstSym)} / protein')




# print('normal:')
	# start = time.time()
	# for i in range(10):
	# 	m = gDodec.buildNodeDistMatrixByLine()
	# print(f'Dur: {time.time()-start}')
	# print(m[:2])
	# print('-- k:')
	# start = time.time()
	# for i in range(10):
	# 	m = gDodec.buildNodeDistMatrixKByLine()
	# print(f'Dur: {time.time()-start}')
	# print(m[:2])
