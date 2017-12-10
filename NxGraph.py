from networkx import *
from py2neo import Graph as DbGraph


class NxProtGraph(Graph):
	def buildFromDb(self, graphDb:DbGraph, pdbEntry:str): bool
	
	
