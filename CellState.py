
class CellState:
	# Don't show these attributes in gui (not used any more?)
	excludeAttr = ['divideFlag']
	excludeAttr = ['deathFlag']

	def __init__(self, cid):
		self.id = cid
		self.growthRate = 1.0
		self.color = [0.5,0.5,0.5]
		self.divideFlag = False
		self.deathFlag = False

