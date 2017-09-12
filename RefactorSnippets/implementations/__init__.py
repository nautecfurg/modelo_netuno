import os
from shape import Shape
import importlib
print("running")


for module in os.listdir(os.path.dirname(__file__)):
	if module == '__init__.py' or module == 'shape.py' or module[-3:] != '.py':
		continue
	print("importando "+module[:-3])
	__import__(module[:-3], locals(), globals())
	# importlib.import_module(module[:-3])

#del module