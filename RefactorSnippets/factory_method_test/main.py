import implementations as imp
import architecture

subclasses = architecture.Architecture.__subclasses__()
subaclasses_dict = {}
for arch in subclasses:
    subaclasses_dict[arch.__name__] = arch
ARCHITECTURE = subaclasses_dict['ArchitectureB']()
ARCHITECTURE.prediction()
