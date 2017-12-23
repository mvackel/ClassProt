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
	
	print(gDodec.buildLine(1,weight=None))
	print(gDodec.buildKLine(1,weight=None))
	print(gDesargues.buildLine(1,weight=None))
	print(gDesargues.buildKLine(1,weight=None))
	print('--------------')
	print(gDodec.buildKLine(1,2,weight=None))
	print(gDesargues.buildKLine(1,2,weight=None))

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
