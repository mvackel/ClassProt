import random
import time

from thespian.actors import *
from thespian.troupe import troupe
from py2neo import Graph, Node, Relationship

from Tools import *

# Se der erro no ActorSystem, para poder utilizar novamente, basta encerrar
# os processos PyThon(32bit) no Task Manager

startTime = time.time()


class Start():  # msg from main sender to Coordinator
	def __init__(self, dbUser: str, dbPwd: str):  # , dbGraph: Graph):
		# self.dbGraph = dbGraph
		self.dbUser = dbUser
		self.dbPwd = dbPwd
		self.startTime = time.time()


class CooStarted():  # msg from Coordinator to main sender
	def __init__(self, errCode):
		self.errCode = errCode
		self.StartedAt = time.time()


class Process():  # msg from main sender to Coordinator
	def __init__(self, pdbList: list):
		self.pdbEntryList = pdbList


class Ended():  # msg from Coordinator to main sender
	def __init__(self, numSymProcessed):
		self.numSymProcessed = numSymProcessed
		self.EndedAt = time.time()


class CooGetPDB():  # msg from Coordinator to GetPDBActor
	def __init__(self, nMsg: int, pdbEntry: str, dbUser:str, dbPwd:str):  # , dbUser:str, dbPwd:str):
		self.nMsg = nMsg
		self.pdbEntry = pdbEntry
		self.dbUser = dbUser
		self.dbPwd = dbPwd
	
	# self.dbUser = dbUser
	# self.dbPwd = dbPwd


class GetPdbEntryEnded():  # msg from StorePDBGraphActor to Coordinator
	def __init__(self, nMsg: int, pdbEntry: str, numNodes:int, numRels: int):
		self.nMsg = nMsg
		self.pdbEntry = pdbEntry
		self.numNodes = numNodes
		self.numRels = numRels  # numero de relacionamentos criados
	
# class GetPdbCAlphaList():  # msg from GetPDBActor to StorePDBGraphActor
# 	def __init__(self, coord, nMsg: int, pdbEntry: str, lstCAlpha: list):
# 		self.coordinator = coord  # return result to coordinator
# 		self.nMsg = nMsg
# 		self.pdbEntry = pdbEntry
# 		self.lstCAlpha = lstCAlpha
#
# class StorePdbEntryEnded():  # msg from StorePDBGraphActor to Coordinator
# 	def __init__(self, nMsg: int, pdbEntry: str, numRels: int):
# 		self.nMsg = nMsg
# 		self.pdbEntry = pdbEntry
# 		self.numRels = numRels  # numero de relacionamentos criados


# ===============   GET PDB ACTOR
@troupe(max_count=50, idle_count=1)
class GetPDBActor(ActorTypeDispatcher):
	def receiveMsg_CooGetPDB(self, getPdbMsg: CooGetPDB, sender):
		self.troupe_work_in_progress = True
		nMsg = getPdbMsg.nMsg
		pdbEntry = getPdbMsg.pdbEntry
		coordinator = sender
		dbUser = getPdbMsg.dbUser
		dbPwd  = getPdbMsg.dbPwd
		print(f"--GetPDB-TROUPE: {nMsg}")
		
		# Pega entrada no PDB (Web):
		# Teste: casFromPdb = [CAlpha('AABB', 10, 'CA', 123, (12, 14, 16))]  # CAlpha(idPdb, resSeq, resName, serial, aCoo)
		(entry, casFromPdb) = getPDB(getPdbMsg.pdbEntry)
		
		# cria ator para armazenar a entrada no DB:
		# dbStoreEntry = self.createActor(GraphDBStorePDB)
		# Armazena no DB os carbonos alfa - sincrono:
		py2neo.authenticate("localhost:7474", dbUser, dbPwd)
		#py2neo.authenticate("localhost:7474", 'neo4j', 'pxyon123')
		dbGraph = Graph()

		nNodes = nRels = 0
		# veririca se o DB ja' tem essa entrada:
		if not existsNodePDB(dbGraph, pdbEntry):
			self.graphDBStoreEntry(dbGraph, entry)
			nNodes, nRels = self.graphDBStoreCAlfas(dbGraph, pdbEntry, casFromPdb)
			print(f"--GetPDB: fim. : {nMsg} Id: {pdbEntry}  #CA: {len(casFromPdb)}  #Rel: {nRels}")
		else:
			print(f"--GetPDB: fim. Ja' existe: {pdbEntry}")

		msgEnded = GetPdbEntryEnded(nMsg, pdbEntry, nNodes, nRels)
		self.send(coordinator, msgEnded )
		self.troupe_work_in_progress = False

	# ==============   STORE PDB Entry
	def graphDBStoreEntry(self, dbGraph:Graph, entry: PDBEntry):
		tx = dbGraph.begin()
		tx.merge(entry)
		tx.commit()

	# ===============   STORE CAlphas
	def graphDBStoreCAlfas(self, dbGraph:Graph, pdbEntry: str, casFromPdb: list) -> (int, int):
		
		# cria os nodos:
		tx = dbGraph.begin()
		for ca in casFromPdb:
			tx.merge(ca)
			#dbGraph.merge(ca)
		tx.commit()
		
		numCAs = len(casFromPdb)
		# print(f'---Store: PDB: {pdbEntry} Numero de Nodos Criados: {numCAs}')

		# cria os relacionamentos:
		numRels = 0
		tx = dbGraph.begin()
		for nca, oCa in enumerate(casFromPdb):
			dists = calcDistances(casFromPdb, nca)
			for n, dist in enumerate(dists[nca:]):  # sempre a partir do CAlpha em diante
				if dist > 0.0 and dist <= 10.0:
					rel = Near_10A(oCa, casFromPdb[nca + n])
					# 					#dbGraph.merge(rel)
					tx.merge(rel)
					numRels += 1
		tx.commit()
		
		return (numCAs, numRels)


# def receiveUnrecognizedMessage(self, message, sender):
# 	print(f'Received Unrecognized Message: {sender} {message}')



# ===============   COORDINATOR ACTOR
from PdbToGraphDB import *


class Coordinator(ActorTypeDispatcher):
	def __init__(self):
		self.started = False
		self.numEntriesToProcess = 0
		self.numEntriesProcessed = 0
		self.mainSender = None
		self.pdbToGraphDB = PdbToGraphDB
		self.graphDB = None
		self.dbUser = "neo4j"
		self.dbPwd = "pxyon123"
		super(Coordinator, self).__init__()
	
	def receiveMsg_Start(self, startMsg: Start, sender):
		self.started = False
		self.mainSender = sender
		self.dbUser = startMsg.dbUser
		self.dbPwd = startMsg.dbPwd
		errCode = -1  # error
		# self.graphDB = startMsg.dbGraph
		# if self.graphDB is not None:
		errCode = 0
		self.started = True
		self.send(self.mainSender, CooStarted(errCode))
	
	def receiveMsg_Process(self, lstPdbEntriesMsg: Process, sender):
		lstEntries: list = lstPdbEntriesMsg.pdbEntryList
		self.numEntriesToProcess = len(lstEntries)
		self.numEntriesProcessed = 0
		# print(self.Started)
		if self.started:
			if self.numEntriesToProcess > 0:
				# cria os atores: (um so' ator, pois e' troupe) -- troupe nao consegue criar actors: Windows Permission Error
				# getPdbActor = self.createActor(GetPDBActor)
				# envia as mensagens:
				for nMsg, sEntry in enumerate(lstEntries):
					getPdbActor = self.createActor(GetPDBActor)
					self.send(getPdbActor, CooGetPDB(nMsg, sEntry, self.dbUser, self.dbPwd))
			else:
				# Nenhuma entrada para processar: Encerrar:
				self.send(sender, Ended(0))
		else:
			# Nao iniciou com Start. Enviar um erro:
			self.send(sender, Ended(-1))
	
	def receiveMsg_GetPdbEntryEnded(self, message: GetPdbEntryEnded, sender):
		print(
			f'-Coo: rec: StorePdbEntryEnded. nMsg: {message.nMsg}')  # print(f'Coord: 2: Received reply message: {num} : {msg}')
		if self.started:
			self.numEntriesProcessed += 1
			if self.numEntriesProcessed == self.numEntriesToProcess:
				# recebeu todas as respostas. Encerrar:
				self.send(self.mainSender, Ended(self.numEntriesProcessed))
			# self.send(sender, (num, f'Received Tuple Msg: {msg}'))
			
			# def receiveUnrecognizedMessage(self, message, sender):
			# 	self.send(sender, 'Coord: Received Unrecognized Message')


def run_example(dbUser: str, dbPwd: str, pdbSymList: list, timeout=10, systembase=None):
	coordinator: Coordinator = ActorSystem(systembase).createActor(Coordinator)
	print(f'---- Start: {time.time()-startTime}')
	msgStart = Start(dbUser, dbPwd)
	started: CooStarted = ActorSystem().ask(coordinator, msgStart, 0.5)
	
	if started is not None:
		ended: Ended = ActorSystem().ask(coordinator, Process(pdbSymList), timeout)
		if ended is not None:
			print("=== FINISHED OK ===")
			print(f"=== Numero de simbolos PDB processados: {ended.numSymProcessed}")
			print("===")
		else:
			print(f"### Timeout ###. {ended}")
	
	else:
		print("### No Start ### : {started}")
	print(f'---- End: {time.time()-startTime}')
	
	ActorSystem(systembase).shutdown()


if __name__ == "__main__":
	# import sys
	
	# pdbEntriesList = ['1LS6:1' '1Z28:1' '2D06:1' '3QVU:1' '3QVV:1' '3U3J:1' '3U3K:1' '3U3M:1' '3U3O:1' '3U3R:1' '4GRA:1' ]
	#lstPdbEntries = ['1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA' ]
	lstPdbEntries = ['1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA']
	# lstPdbEntries = [ '5O75', '1LS6', '1Z28', '2D06', '3QVU', '3U3J' ]
	# lstPdbEntries = ['5O75', '1LS6', '5NH5', '1Z28', '5WEA']
	lstPdbEntries = ['5O75', '1G8P', '2X31', '5EWU']
	# py2neo.authenticate("localhost:7474", "neo4j", "pxyon123")
	# dbGraph = Graph()
	# 'multiprocTCPBase'
	# 'multiprocUDPBase'
	# 'simpleSystemBase' == None
	# run_example(sys.argv[1] if len(sys.argv) > 1 else None)
	# run_example()
	dbUser = "neo4j"
	dbPwd = "pxyon123"
	segsPerEntry = 6     # esperado: 4 segundos por entrada
	timeout = len(lstPdbEntries) * segsPerEntry
	run_example(dbUser, dbPwd, lstPdbEntries, timeout, 'multiprocTCPBase')
# run_example(dbUser, dbPwd, lstPdbEntries, 'multiprocUDPBase')
# run_example(dbUser, dbPwd, lstPdbEntries, 'simpleSystemBase')

# MATCH (c)-[n:NEAR_10A]->() WHERE n.Dist <= 7 RETURN c,n
# MATCH (c)-[n:NEAR_10A]->() WHERE n.Dist <= 7 and c.IdPDB="5O75" RETURN c,n
