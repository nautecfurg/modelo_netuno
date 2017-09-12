import random
import implementations as imp
# import shape_implementations as si

# # Generate shape name strings:
# def shapeNameGen(n):
# 	types = imp.Shape.__subclasses__()
	
	
# 	print(types)
	
# 	# exit('DIE MODAFUUKA DIE')
# 	for i in range(n):
# 		rtype = random.choice(types)
# 		yield rtype.__name__

# shapes = \
# [ imp.Shape.factory(i) for i in shapeNameGen(7)]

shapes = []
shapes += [imp.Shape.__subclasses__()[0]()]
shapes += [imp.Shape.__subclasses__()[1]()]

for shape in shapes:
	shape.draw()
	shape.erase()