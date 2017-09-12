import os
import importlib

for module in os.listdir(os.path.dirname(__file__)):

    if module != '__init__.py' and module[-3:] == '.py':
        __import__('Optimizers.'+module[:-3], locals(), globals())

    elif os.path.isdir(os.path.dirname(__file__)+'/'+module):
        abs_module = os.path.dirname(__file__)+'/'+module
        for submodule in os.listdir(abs_module):
            if submodule != '__init__.py' and submodule[-3:] == '.py':
                __import__('Optimizers.' + module +
                           '.' + submodule[:-3], locals(), globals())
