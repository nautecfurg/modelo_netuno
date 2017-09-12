import os
import importlib

for module in os.listdir(os.path.dirname(__file__)):
    if module != '__init__.py' and module[-3:] == '.py':
        __import__('implementations.'+module[:-3], locals(), globals())

