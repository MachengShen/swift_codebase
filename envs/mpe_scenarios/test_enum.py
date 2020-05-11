from enum import Enum, unique

@unique
class CellLocation(Enum):
	#Enumeration of celllocations within a room
	UpperLeft = 1
	UpperRight = 2
	BottomLeft = 3
	BottomRight = 4

print(CellLocation.BottomRight)