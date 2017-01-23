import functools
import operator as op
from collections import defaultdict

def vectorize(f):
    def wrapper(obj, *args):
        bound = functools.partial(f, obj)
        return [bound(a) for a in args]
    return wrapper


class SimpleTree( object ):
    def __init__(self, nodes, 
                 node_id_cb, 
                 parent_id_cb):

        self._nodes = defaultdict(lambda *x: None, { node_id_cb(n):n for n in nodes })
        self._parent_ids = defaultdict(lambda *x: None, { nid:parent_id_cb(n) for nid,n in self._nodes.iteritems() })
        self._child_ids = defaultdict(lambda *x: None, { nid:[] for nid in self._nodes })

        for nid in self._parent_ids:
            pid = self._parent_ids[nid]
            if pid is not None:
                self._child_ids[pid].append(nid)


    def filter_nodes(self, criterion):
        return filter(criterion, self._nodes.values())
        
        
    def value_map(self, from_fn, to_fn):
        return {from_fn(v): to_fn(v) for v in self._nodes.values()}


    def node_ids(self):
        return self._nodes.keys()


    @vectorize
    def parent_id(self, nid):
        return self._parent_ids[nid]

   
    @vectorize
    def child_ids(self, nid):
        return self._child_ids[nid]


    @vectorize
    def ancestor_ids(self, nid):
        pid = self.parent_id(nid)[0]
        if pid is not None: # required to avoid failing on 0
            return self.ancestor_ids(pid)[0] + [nid]
        else:
            return [nid]
            
    
    @vectorize
    def descendant_ids(self, nid):
        children = self.child_ids(nid)[0]
        if children:
            return [nid] + reduce(op.add, self.descendant_ids(*children))
        return [nid]
            

    @vectorize
    def node(self, nid):
        return self._nodes[nid]


    @vectorize
    def parent(self, nid):
        return self.node(self.parent_id(nid)[0])


    @vectorize
    def children(self, nid):
        return self.node(*self.child_ids(nid)[0])


    @vectorize
    def descendants(self, nid):
        return self.node(*self.descendant_ids(nid)[0])

    
    @vectorize
    def ancestors(self, nid):
        return self.node(*self.ancestor_ids(nid)[0])
        
        
class SimpleTreeWithLists(SimpleTree):


    def parent_id(self, nids):
        return [ self._parent_ids[nid] for nid in nids ]

   
    def child_ids(self, nids):
        return [ self._child_ids[nid] for nid in nids ]


    def ancestor_ids(self, nids):
        for nid in nids:
        
            pid = self.parent_id([nid])[0]
            if pid:
                return self.ancestor_ids([pid])[0] + [nid]
            else:
                return [nid]
            
    
    def descendant_ids(self, nids):
        for nid in nids:
            children = self.child_ids([nid])[0]
            if children:
                return [nid] + reduce(op.add, self.descendant_ids(children))
            return [nid]
            

    def node(self, nids):
        return [ self._nodes[nid] for nid in nids ]


    def parent(self, nids):
        return map(self.node, map(self.parent_id, nids))


    def children(self, nids):
        return [ map(self.node, self.child_ids(nid)) for nid in nids ]


    def descendants(self, nids):
        return [ map(self.node, self.descendant_ids(nid)) for nid in nids ]

    
    def ancestors(self, nid):
        return [ map(self.node, self.ancestor_ids(nid)) for nid in nids ]
    
