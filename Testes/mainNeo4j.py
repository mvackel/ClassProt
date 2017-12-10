

# delete all:
#MATCH (n)
#DETACH DELETE n

from py2neo import Graph
import py2neo


#py2neo.authenticate("localhost:7474/db/data","neo4j","pxyon123")
py2neo.authenticate("localhost:7474","neo4j","pxyon123")

graph = Graph()

#import cypher
#cypher.Connection()
#aa = cypher.run("MATCH (n:Person)-[KNOWS]-(b) RETURN a, b")
#print( aa )

query = """
MATCH (person:Person)-[:KNOWS]->(p:Person)
RETURN person.name AS name, p.name AS known
"""

data = graph.run(query)

for d in data:
    print(d)

query = """
CREATE (p: Person :PDB :Enzime {name: 'Marcos', code: '13.1.1.1'})
"""

data = graph.run(query)

for d in data:
    print(d)


