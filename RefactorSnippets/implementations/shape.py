
class Shape(object):
	typeDict = {"Circle": circle.Circle, "Square": square.Square}
	# Create based on class name:
	# @static
	def factory(type):
		# print('tipo escolhido '+ type)
		#return eval(type + "()")
		#if type == "Circle": return circle.Circle()
		#if type == "Square": return square.Square()
		#assert 0, "Bad shape creation: " + type
		try:
			return typeDict[type]()
		except KeyError as err:
			raise ValueError("%s: Invalid type" % err)
	factory = staticmethod(factory)
