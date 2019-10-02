class SafeJsonMsg:
    ''' Apes a paged query response from api.brain-map.org. 
        Safe to use with Pythons >= 3.7 (which implement pep 479, such that StopIteration errors in 
        generators are converted to RunTimeErrors).
    '''

    def __init__(self, data):
        self.data = iter(data)

    def __call__(self, *a, **k):
        try:
            return next(self.data)
        except StopIteration as err:
            return {'msg': []}
