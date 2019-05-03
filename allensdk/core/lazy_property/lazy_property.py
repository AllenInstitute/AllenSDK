class LazyProperty(object):
    
    def __init__(self, api_method, wrappers=tuple(), cache=None, *args, **kwargs):

        self.api_method = api_method
        self.wrappers = wrappers
        self.args = args
        self.kwargs = kwargs
        self.value = None
        self.cache = cache

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        if self.value is None:
            self.value = self.calculate()
        return self.value

    def __set__(self, obj, value):
        raise AttributeError("Can't set LazyLoadable attribute")

    def calculate(self):

        a, b = self.api_method.__module__, self.api_method.__name__
        def f(self):
            return '.'.join([a, b, str(self.get_ophys_experiment_id())])

        result = self.api_method(*self.args, get_key=f, **self.kwargs)
        for wrapper in self.wrappers:
            result = wrapper(result)
        return result
