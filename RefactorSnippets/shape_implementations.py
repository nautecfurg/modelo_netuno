from  shape import Shape

class Circle(Shape):
	def draw(self): print("Circle.draw")
	def erase(self): print("Circle.erase")

class Square(Shape):
	def draw(self): print("Square.draw")
	def erase(self): print("Square.erase")
