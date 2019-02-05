class LazyProperty(object):
    
    def __init__(self, api_method):
        self.getter_name = api_method
        self.hidden_attr = '__' + self.getter_name + '_retval'

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        if not hasattr(obj, self.hidden_attr):
            setattr(obj, self.hidden_attr, self.calculate(obj))
        return getattr(obj, self.hidden_attr)

    def __set__(self, obj, value):
        raise AttributeError("Can't set LazyLoadable attribute")

    def __delete__(self, obj):
        del self.cache[obj]

    def calculate(self, obj):
        return getattr(obj.api, self.getter_name)()