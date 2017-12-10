
import requests
import urllib
from urllib import parse



#http = urllib3.PoolManager()

url = 'https://www.rcsb.org/pdb/rest/search/'

# queryText = """
# 			<?xml version="1.0" encoding="UTF-8"?>
# 			<orgPdbQuery>
# 			<version>B0907</version>
# 			<queryType>org.pdb.query.simple.ExpTypeQuery</queryType>
# 			<description>Experimental Method Search: Experimental Method=SOLID-STATE NMR</description>
# 			<mvStructure.expMethod.value>SOLID-STATE NMR</mvStructure.expMethod.value>
# 			</orgPdbQuery>
# 			"""

# queryText = """
#     <orgPdbQuery>
#         <queryType>org.pdb.query.simple.UpAccessionIdQuery</queryType>
#         <description>Simple query for a list of UniprotKB Accession IDs: P50225</description>
#         <accessionIdList>P50225</accessionIdList>
#     </orgPdbQuery>
# """

# queryText = """
#   <orgPdbQuery>
#     <version>head</version>
#     <queryType>org.pdb.query.simple.TreeEntityQuery</queryType>
#     <description>EnzymeClassificationTree Search for 6: Ligases</description>
#     <queryId>D5FB6FDC</queryId>
#     <resultCount>2968</resultCount>
#     <runtimeStart>2017-11-19T00:09:25Z</runtimeStart>
#     <runtimeMilliseconds>21</runtimeMilliseconds>
#     <t>3</t>
#     <n>7309</n>
#     <nodeDesc>6: Ligases</nodeDesc>
#   </orgPdbQuery>
#   """

# queryText = """
# <orgPdbCompositeQuery version="1.0">
#     <resultCount>37810</resultCount>
#     <queryId>2A1D936F</queryId>
#  <queryRefinement>
#   <queryRefinementLevel>0</queryRefinementLevel>
#   <orgPdbQuery>
#     <version>head</version>
#     <queryType>org.pdb.query.simple.TreeQueryExpression</queryType>
#     <description>TAXONOMY is just Homo sapiens (human)</description>
#     <queryId>FFE58E6C</queryId>
#     <resultCount>37810</resultCount>
#     <runtimeStart>2017-11-19T00:12:23Z</runtimeStart>
#     <runtimeMilliseconds>427</runtimeMilliseconds>
#     <t>1</t>
#     <n>+9606</n>
#     <nodeDesc>Expression using Homo sapiens (human)</nodeDesc>
#   </orgPdbQuery>
#  </queryRefinement>
# </orgPdbCompositeQuery>
# """
# Query copiado do site do PDB (www.rcsb.org). Fazer uma "advanced query".
# Clicar no botao "Query Details" na tela de resultados da quary - Lado esquerdo, no fim da secao.

# queryText = """
# <orgPdbCompositeQuery version="1.0">
#     <resultCount>37810</resultCount>
#     <queryId>2A1D936F</queryId>
#  <queryRefinement>
#   <queryRefinementLevel>0</queryRefinementLevel>
#   <orgPdbQuery>
#     <version>head</version>
#     <queryType>org.pdb.query.simple.TreeQueryExpression</queryType>
#     <description>TAXONOMY is just Homo sapiens (human)</description>
#     <queryId>FFE58E6C</queryId>
#     <resultCount>37810</resultCount>
#     <runtimeStart>2017-11-19T00:12:23Z</runtimeStart>
#     <runtimeMilliseconds>427</runtimeMilliseconds>
#     <t>1</t>
#     <n>+9606</n>
#     <nodeDesc>Expression using Homo sapiens (human)</nodeDesc>
#   </orgPdbQuery>
#  </queryRefinement>
#  <queryRefinement>
#   <queryRefinementLevel>1</queryRefinementLevel>
#   <conjunctionType>and</conjunctionType>
#   <orgPdbQuery>
#     <version>head</version>
#     <queryType>org.pdb.query.simple.EnzymeClassificationQuery</queryType>
#     <description>Enzyme Classification Search : EC=6.5.*</description>
#     <queryId>B0F548BF</queryId>
#     <resultCount>102</resultCount>
#     <runtimeStart>2017-11-19T00:12:24Z</runtimeStart>
#     <runtimeMilliseconds>3</runtimeMilliseconds>
#     <Enzyme_Classification>6.5.*</Enzyme_Classification>
#   </orgPdbQuery>
#  </queryRefinement>
# </orgPdbCompositeQuery>
# """


queryText = """
 <orgPdbQuery>
    <version>head</version>
    <queryType>org.pdb.query.simple.TreeEntityQuery</queryType>
    <description>EnzymeClassificationTree Search for 6.6: Forming nitrogen-metal bonds</description>
    <queryId>239D25EE</queryId>
    <resultCount>4</resultCount>
    <runtimeStart>2017-11-26T00:20:20Z</runtimeStart>
    <runtimeMilliseconds>0</runtimeMilliseconds>
    <t>3</t>
    <n>7546</n>
    <nodeDesc>6.6: Forming nitrogen-metal bonds</nodeDesc>
  </orgPdbQuery>
"""

print ("query:\n", queryText)
print ("querying PDB...\n")

#headers = {'Content-Type': 'application/xml'}
headers = {'Content-Type': 'application/x-www-form-urlencoded'}

#queryText = queryText.encode()
queryText = urllib.parse.quote(queryText, safe='')

req = requests.post(url, data=queryText, headers=headers)


#f = urllib2.urlopen(req)
#result = f.read()
result = req.text

if result:
	print ( result )
	print ("Found number of PDB entries:", result.count('\n'))
else:
	print ("Failed to retrieve results")