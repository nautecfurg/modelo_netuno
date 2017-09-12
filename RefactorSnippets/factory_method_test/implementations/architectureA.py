import sys
sys.path.insert(0,'..')
import architecture
class ArchitectureA(architecture.Architecture):
    def prediction(self):
        print("A faz a predicao usando A")
    