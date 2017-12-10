import py2neo
from py2neo import Graph

from Tools import *
from thespian.actors import *
from PDBActors import *

class PdbToGraphDB:
	def __init__(self, dbUser, dbPwd):
		self.dbUser = dbUser
		self.dbPwd = dbPwd
		self.dbGraph = None
		self.systembase = 'simpleSystemBase'
		
	def setSytemBase(self, systembase=None):
		# 'multiprocTCPBase'
		# 'multiprocUDPBase'
		# 'simpleSystemBase' == None
		self.systembase = systembase
		#TODO: testar systembase string
		
	def initGraphDB( self ) -> Graph :
		py2neo.authenticate("localhost:7474", self.dbUser, self.dbPwd)
		self.dbGraph = Graph()
		return self.dbGraph
	
	def createFromPdbList(self, pdbEntries: list, timeout:int):
		coordinator: Coordinator = ActorSystem(self.systembase).createActor(Coordinator)
		print(f'-- Start: {time.time()-startTime}')
		started: CooStarted = ActorSystem().ask(coordinator, Start(self.dbUser, self.dbPwd), 0.5)
		
		if started is not None:
			ended: Ended = ActorSystem().ask(coordinator, Process(pdbEntries), timeout)
			if ended is not None:
				print("=== FINISHED OK ===")
				print(f"=== Numero de simbolos PDB processados: {ended.numSymProcessed}")
				print("===")
			else:
				print(f"### Timeout ###. {ended}")
		
		else:
			print("### No Start ### : {started}")
		print(f'---- End: {time.time()-startTime}')
		
		ActorSystem(self.systembase).shutdown()
	
	def createFromPdb( self, pdbId: str ) -> int :
		self.initGraphDB()
		casFromPdb = getPDB( pdbId )
	
		# cria os nodos:
		for ca in casFromPdb :
			self.dbGraph.merge( ca )
	
		numCAs = len(casFromPdb)
		print(f'PDB: {pdbId} Numero de Nodos Criados: {numCAs}')
	
		# cria os relacionamentos:
		numRels = 0

		for nca, oCa in enumerate( casFromPdb ) :
			dists = calcDistances( casFromPdb, nca )
			for n, dist in enumerate(dists[nca:]):      # sempre a partir do CAlpha em diante
				if dist > 0.0 and dist <= 10.0 :
					rel = Near_10A(oCa, casFromPdb[nca+n])
					self.dbGraph.merge(rel)
					numRels += 1
	
		print(f'PDB: {pdbId} Numero de Relacionamentos Criados: {numRels}')
		return numRels


if __name__ == "__main__":
	
	pdbToGraphDB = PdbToGraphDB("neo4j", "pxyon123")
	serialSys = 'simpleSystemBase'
	parallelSys = 'multiprocTCPBase'
	
	# Para Testes: Mudar aqui para paralelo ou serial
	pdbToGraphDB.setSytemBase( parallelSys )

	# Teste com apenas 1 entrada PDB:
	#pPdbToGraphDB.createFromPdb("5O75")

	# Teste com uma lista de entradas PDB:
	lstPdbEntries = ['1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA']
	segsPerEntry = 4     # esperado: 4 segundos por entrada
	timeout = len(lstPdbEntries) * segsPerEntry
	pdbToGraphDB.createFromPdbList( lstPdbEntries, timeout)







