import sys
def get_implementation(parent_class, child_class_name):
    """Returns a subclass instance.
    It searchs in the subclasses of `parent_class` a class
    named `child_class_name` and return a instance od this class
    Args:
        parent_class: parent class.
        child_class_name: string containing the child class name.
    Returns:
        child_class: instance of child class. `None` if not found.
    """
    for child_class in parent_class.__subclasses__():
        if child_class.__name__ == child_class_name:
            return child_class()
    return None

def is_subclass(parent_class, child_class_name):
    """Checks if the parent class has a child with a given name.

    It searchs in the subclasses of `parent_class` a class
    named `child_class_name`. Return True if found and False if not found.

    Args:
        parent_class: parent class.
        child_class_name: string containing the child class name.

    Returns:
        True if found and False if not found.
    """
    for child_class in parent_class.__subclasses__():
        if child_class.__name__ == child_class_name:
            return True
    return False


def arg_validation(arg, cla):
    """
    Checks if the argument corresponds to a valid class.

    This function checks in the subclasses of `cla` if a class with name equal to `arg` exists.
    If `arg` is a name of a subclass it returns the arg. If `arg` is not found it shows a
    a error message.


    Args:
        arg: child class name.
        cla: parent class.

    Returns:
        arg: child class name.
    """
    if is_subclass(cla, arg):
        return arg
    else:
        print(str(arg)+" is not a valid " + cla.__module__ + " name.")
        sys.exit(2)
