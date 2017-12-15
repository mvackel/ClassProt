import sys
import requests
import urllib
from urllib import parse
import os
from PdbToGraphDB import *

from PDBActors import *

class DBGraphFromPDB:
	def __init__(self):
		self.url = 	'https://www.rcsb.org/pdb/rest/search/'

	def run(self, parms) -> int:
		(self.queryFName, self.dbUser, self.dbPwd) =  self.VerifyParms(parms)
		print(f'Parameters: {self.queryFName}, {self.dbUser}, {self.dbPwd}')
		print(f'File: {os.getcwd()}\{self.queryFName}')
		# Carrega o arquivo com a query para o PDB (usar "advanced search")
		# Doc: https://www.rcsb.org/pdb/software/rest.do#search
		queryText, err = self.GetQueryFileText( self.queryFName )
		if err < 0:
			print('File not found.' if err == -1 else 'Error Reading file.')
			return 1
		if len(queryText) == 0:
			print(f'File is empty.')
			return 2
		# Executar a query ao PDB:
		lstPdbEntries :list = self.QueryPDB( queryText )
		print( f'Found number of PDB entries: {len(lstPdbEntries)}' )
		if len(lstPdbEntries) == 0:
			print('Exiting...')
			return 3
		# Carregar os resultados no Neo4j (DB Grafico):
		#err = self.Load
		
		segsPerEntry = 6  # esperado: 4 segundos por entrada
		timeout = len(lstPdbEntries) * segsPerEntry
		run_example(self.dbUser, self.dbPwd, lstPdbEntries, timeout, 'multiprocTCPBase')
	
	
	def VerifyParms(self, parms):
		queryFName = dbUser = dbPwd = ''
		if len(parms) == 4:
			dbPwd = parms[3]
			dbUser = parms[2]
			queryFName = parms[1]
		
		if len(parms) <= 3:
			dbPwd = 'pyxon123'
		if len(parms) <= 2:
			dbUser = 'neo4j'
		if len(parms) <= 1:
			queryFName = "PDBQuery.xml"
		return (queryFName, dbUser, dbPwd)

	def GetQueryFileText(self, filename :str) -> tuple:
		#print(os.getcwd())
		err = -1   # file not found
		text = ''
		if os.path.exists(filename):
			err = 0     # ok
			with open(filename, 'r') as file:
				try:
					text = file.read()
				except:
					err = -2    # read error
		return (text, err)

	def QueryPDB(self, queryText :str) -> list:
		if len(queryText) > 0:
			#print("query:\n", queryText)
			print("Querying PDB...\n")
			
			# headers = {'Content-Type': 'application/xml'}
			headers = {'Content-Type': 'application/x-www-form-urlencoded'}
			
			queryText = urllib.parse.quote(queryText, safe='')
			
			req = requests.post(self.url, data=queryText, headers=headers)
			reqTxt = req.text
			
			result = []
			if reqTxt:
				import re
				a = re.split(r'[:\n]+', reqTxt)
				result :list = [ x for x in re.split(r'[:\n]+', reqTxt) if len(x)==4 ]
				#print("Found number of PDB entries:", result.count('\n'))
				return result
	
	def createGraphFromPDB(self, pdbEntries: list):
		dbWriter = PdbToGraphDB(self.dbUser, self.dbPwd)
		dbWriter.createFromPdbList(pdbEntries)
			

if __name__ == "__main__":
	a = sys.argv
	DBGraphFromPDB().run(sys.argv)
