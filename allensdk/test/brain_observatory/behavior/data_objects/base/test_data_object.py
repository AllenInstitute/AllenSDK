import pytest

from allensdk.brain_observatory.behavior.data_objects import DataObject


class TestDataObject:
    def test_to_dict_simple(self):
        class Simple(DataObject):
            def __init__(self):
                super().__init__(name='simple', value=1)
        s = Simple()
        assert s.to_dict() == {'simple': 1}

    def test_to_dict_nested(self):
        class B(DataObject):
            def __init__(self):
                super().__init__(name='b', value='!')

        class A(DataObject):
            def __init__(self, b: B):
                super().__init__(name='a', value=self)
                self._b = b

            @property
            def prop1(self):
                return self._b

            @property
            def prop2(self):
                return '@'
        a = A(b=B())
        assert a.to_dict() == {'a': {'b': '!', 'prop2': '@'}}

    def test_to_dict_double_nested(self):
        class C(DataObject):
            def __init__(self):
                super().__init__(name='c', value='!!!')

        class B(DataObject):
            def __init__(self, c: C):
                super().__init__(name='b', value=self)
                self._c = c

            @property
            def prop1(self):
                return self._c

            @property
            def prop2(self):
                return '!!'

        class A(DataObject):
            def __init__(self, b: B):
                super().__init__(name='a', value=self)
                self._b = b

            @property
            def prop1(self):
                return self._b

            @property
            def prop2(self):
                return '@'

        a = A(b=B(c=C()))
        assert a.to_dict() == {'a': {'b': {'c': '!!!', 'prop2': '!!'},
                                     'prop2': '@'}}

    def test_not_equals(self):
        s1 = DataObject(name='s1', value=1)
        s2 = DataObject(name='s1', value='1')
        assert s1 != s2

    def test_exclude_equals(self):
        s1 = DataObject(name='s1', value=1, exclude_from_equals={'s1'})
        s2 = DataObject(name='s1', value='1')
        assert s1 == s2

    def test_cannot_compare(self):
        with pytest.raises(NotImplementedError):
            assert DataObject(name='foo', value=1) == 1
