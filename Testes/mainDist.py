
import numpy as ny

# class Position:
#     def __init__(self, aCoord):
#         self.coord = aCoord
#     def dist(self, aNewCoord):
#         return ny.linalg.norm(aNewCoord - self.coord)

# def calcDistances( aAllCoords, idxCAlpha ) :
#     posCAlpha = Position( aAllCoords[idxCAlpha] )
#     dists = []
#     for aCoord in aAllCoords:
#         dists.append( posCAlpha.dist(aCoord) )
#     return dists

def calcDistances( aAllCoords, idxCAlpha) :
    posCAlpha = aAllCoords[idxCAlpha]
    dists = []
    for aCoord in aAllCoords:
        dists.append( ny.linalg.norm( aCoord - posCAlpha ) )
    return dists


def main():
    #a1 = ny.array((1,2,3))
    aCoords = [ny.array((1,2,3)), ny.array((4,5,6)), ny.array((5,6,7))]
    d = calcDistances( aCoords, 1)
    print( d )

#main()

#a = numpy.array((xa ,ya, za))
#b = numpy.array((xb, yb, zb))
#dist = numpy.linalg.norm(a-b)

