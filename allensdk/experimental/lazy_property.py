class LazyProperty(object):
    
    def __init__(self, api_method):
        self.getter_name = api_method
        self.hidden_attr = '__' + self.getter_name + '_retval'

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        if not hasattr(obj.api, self.getter_name):
            raise NotImplementedError("API class {} does not support {}".format(obj.api.__class__.__name__, self.getter_name))

        if not hasattr(obj, self.hidden_attr):
            setattr(obj, self.hidden_attr, self.calculate(obj))
        return getattr(obj, self.hidden_attr)

    def __set__(self, obj, value):
        raise AttributeError("Can't set LazyLoadable attribute")

    def __delete__(self, obj):
        del self.cache[obj]

    def calculate(self, obj):
        return getattr(obj.api, self.getter_name)(obj)

def test_lazy_property():

    invalid_api_method_name = 'get_y'

    class TmpAPI(object):

        def __init__(self, x):
            self.x = x

        def get_x(self, *args, **kwargs):
            return self.x

    class Foo(object):

        x = LazyProperty('get_x')
        y = LazyProperty(invalid_api_method_name)

        def __init__(self, api):
            self.api = api

    x = 5
    api = TmpAPI(x)
    foo = Foo(api)
    assert foo.x == x

    try:
        _ = foo.y
    except NotImplementedError as e:
        assert str(e) == 'API class TmpAPI does not support {}'.format(invalid_api_method_name)

if __name__ == "__main__":
    test_lazy_property()