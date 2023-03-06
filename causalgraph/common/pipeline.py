"""
Pipeline class to define and run several execution steps.
(C) J. Renero, 2023
"""
import types
from typing import List, Any
from tqdm.auto import tqdm


class SampleClass:
    def __init__(self, param1=None, param2=False):
        if param1 is not None:
            self.param1 = param1
            self.param2 = param2

    def fit(self):
        # print(f"  Into the fit of class {self.__class__}")
        # print(f"    I got params: {self.param1}, {self.param2}")
        return ("MyClass.fit")

    def method(self):
        # print(f"  Into the method {self.method.__name__}")
        return ("MyClass.method")


def my_method(param1, param2):
    # print(f"  Into the method {my_method.__name__}")
    # print(f"    I got params: {param1}, {param2}")
    return f"my_method({param1} {param2})"


def m1(what):
    return (f"m1: {what}")


def m2(what):
    return (f"m2: {what}")


m1_message = "(message for m1)"


class Host:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def host_method(self):
        # print(f"  Into the method {self.host_method.__name__}")
        return ("Host.host_method")

# TODO: Eliminate the need to pass the host object to the pipeline
class Pipeline:
    """
    Pipeline class allows to define several execution steps to run sequentially.
    Pipeline is initialized with a host object that contains the parameters to be
    used in the execution steps.
    At each step, the pipeline can call either a function or a class. If a class
    is called, the pipeline will call the fit method of the class. It it's a function,
    it must be present globally or inside the host object.
    The pipeline can also create an attribute in the host object with the value
    returned by the function or the fit method of the class.

    Example:
    --------
    >>> class Host:
    >>>     def __init__(self, param1, param2):
    >>>         self.param1 = param1
    >>>         self.param2 = param2
    >>> 
    >>> def method1(param1, param2):
    >>>     return f"{param1}, {param2}"
    >>> 
    >>> def method2(result1):
    >>>     print(result1)
    >>> 
    >>> 
    >>> host = Host(param1='Hello', param2='world')
    >>> pipeline = Pipeline(host)
    >>> steps = {
    >>>     ('result1', method1): ['param1', 'param2'],
    >>>     method2: ['result1']
    >>> }
    >>>
    >>> pipeline.run(steps)
    >>> print(host.result1)
    >>> Hello, world
    >>> Hello, world
    """

    def __init__(
            self, 
            host: type, 
            prog_bar: bool = True, 
            verbose: bool = False):
        """
        Parameters
        ----------
        host: object
            Object containing the parameters to be used in the execution steps.
        """
        self.host = host
        self._verbose = verbose
        self._prog_bar = prog_bar
        self._objects = [self.host]

    def _get_params(self, param_names) -> List[Any]:
        """
        Get the parameters from the host object.

        Parameters
        ----------
        param_names: list
            List of parameter names to get from the host object.

        Returns
        -------
        params: list
            List of parameters from the host object.
        """
        params = []
        for arg in param_names:
            if type(arg) is str:
                if hasattr(self.host, arg):
                    param = getattr(self.host, arg)
                elif hasattr(self, arg):
                    param = getattr(self, arg)
                elif arg in globals():
                    param = globals()[arg]
                else:
                    raise ValueError(
                        f"Parameter {arg} not found in host object or globals")
            else:
                param = arg
            params.append(param)
        return params

    def run_step(self, step_name: str, list_of_params: List[Any] = []) -> Any:
        """
        Run a step.

        Parameters
        ----------
        step_name: str
            Function or class to be called.
        list_of_params: list
            List of parameters to be passed to the function or class.

        Returns
        -------
        return_value: any
            Value returned by the function or the fit method of the class.
        """
        return_value = None
        # Check if step_name is a function or a class already in globals
        if step_name in globals():
            step_name = globals()[step_name]
            # check if type of step_name is a function
            if type(step_name) is types.FunctionType or type(step_name) is types.MethodType:
                return_value = step_name(*list_of_params)
            # check if type of step_name is a class
            elif type(step_name) is type:
                obj = step_name(*list_of_params)
                obj.fit()
                self._objects.append(obj)
                return_value = obj
            else:
                raise TypeError("step_name must be a class or a function")
        # Check if step_name is a method of the host object
        elif hasattr(self.host, step_name):
            step_name = getattr(self.host, step_name)
            # check if type of step_name is a function
            if type(step_name) is types.FunctionType or type(step_name) is types.MethodType:
                return_value = step_name(*list_of_params)
            else:
                raise TypeError(
                    "step_name inside host object must be a function")
        # Consider that step_name is a method of some of the intermediate objects
        # in the pipeline
        else:
            # check if step name is of the form object.method
            if '.' not in step_name:
                raise ValueError(
                    "step_name must be method of an object: object.method")
            method_call = step_name
            root_object = self.host
            while '.' in method_call:
                obj_name, method_name = step_name.split('.')
                obj = getattr(root_object, obj_name)
                call_name = getattr(obj, method_name)
                method_call = '.'.join(method_name.split('.')[1:])
            return_value = call_name(*list_of_params)

        print("  > Return value:", return_value) if self._verbose else None
        return return_value

    def run(self, steps: dict, desc: str = "Running pipeline"):
        """
        Run the pipeline.

        Parameters
        ----------
        steps: dict
            Dictionary containing the steps to be run. Each key can be a tuple
            containing the name of the attribute to be created in the host object
            and the function or class to be called. But also, each key can be
            a function or method name. In the case of a tuple, the value returned by
            the function or the fit method of the class will be assigned to the
            attribute of the host object. In the case of a function or method name,
            the value returned by the function or the fit method of the class will
            not be assigned to any attribute of the host object.
            The value of each key is a list of parameters to be passed to the 
            function or class. Each parameter must be a string corresponding to
            an attribute of the host object or a value.
        """
        self._pbar = tqdm(total=len(steps.keys()),
                          desc=f"{desc:<25}",
                          leave=False,
                          disable=not self._prog_bar)
        self._pbar.update(0)
        print("-"*80) if self._verbose else None

        for step_name in steps.keys():
            step_params = steps[step_name]
            print(f"Running step {step_name} with params {step_params}") \
                if self._verbose else None
            # check if step is a tuple
            if type(step_name) is tuple:
                vble_name, step_call = step_name
                # create an attribute of name `name`in `self` with the value
                # returned by the function `run_step`
                value = self.run_step(step_call, self._get_params(step_params))
                setattr(self.host, vble_name, value)
                print(f"      New attribute {vble_name}: {getattr(self.host, vble_name)}") \
                    if self._verbose else None
            else:
                self.run_step(step_name, self._get_params(step_params))
            print("-"*80) if self._verbose else None
            self._pbar_update(1)
        self._pbar.close()

    def _pbar_update(self, step=1):
        self._pbar.update(step)
        import time
        time.sleep(1)
        self._pbar.refresh()


if __name__ == "__main__":
    host = Host('value1', 'value2')
    pipeline = Pipeline(host)
    steps = {
        ('myobject', 'SampleClass'): ['param1', True],
        'myobject.method': [],
        'host_method': [],
        'my_method': ['param1', 'param2'],
        ('r1', 'm1'): ['m1_message'],
        ('r2', 'm2'): ['r1']
    }

    pipeline.run(steps)