API Access
==========

The :py:mod:`allensdk.api` package is designed to help retrieve data from the
`Allen Brain Atlas API <http://help.brain-map.org/display/api/Allen+Brain+Atlas+API>`_. :py:mod:`~allensdk.api`
contains methods to help formulate API queries and parse the returned results.  There are several
pre-made subclasses available that provide pre-made queries specific to certain data sets. Currently there 
are three subclasses in Allen SDK:

    * :py:class:`~allensdk.api.queries.cell_types_api.CellTypesApi`: data related to the Allen Cell Types Database  
    * :py:class:`~allensdk.api.queries.biophysical_perisomatic_api.BiophysicalPerisomaticApi`: data related to perisomatic biophysical models
    * :py:class:`~allensdk.api.queries.glif_api.GlifApi`: data related to GLIF models


Creating New API Query Classes
------------------------------

The following example demonstrates how to create a subclass of the generic
:py:class:`~allensdk.api.api.Api` class
in order to generate new queries and read data from the structure of
the response json.
The new class uses :py:meth:`~allensdk.api.api.Api.do_rma_query` to execute the query.

For example, to download gene metadata from the API, first create gene_acronym_query.py::

    from allensdk.api.api import Api
    
    def GeneAcronymQuery(Api):
        def __init__(self):
            super(GeneAcronymQuery, self).__init__()
            
        def build_rma(self, acronym):
            '''Compose a query url'''
            
            return ''.join([self.rma_endpoint,
                           "/Gene/query.json",
                           "?criteria=",
                           "[acronym$il'%s']" % (acronym),
                           "&include=organism",
                           ])
        
        def read_json(self, json_parsed_data):
            '''read data from the result message'''
            
            if 'msg' in json_parsed_data:
                return json_parsed_data['msg']
            else:
                raise Exception("no message!")
        
        def get_data(self, acronym):
            '''Use do_rma_query() from the Api class to execute the query.'''
            return self.do_rma_query(self.build_rma,
                                     self.read_json,
                                     acronym)

The query class is then simple to use in a script.  Create main.py::

    from gene_acronym_query import GeneAcronymQuery
    
    query = GeneAcronymQuery()
    gene_info = query.get_data('ABAT')
    for gene in gene_info:
        print "%s (%s)" % (gene['name'], gene['organism']['name'])

Additional documentation is available to help with
`constructing an RMA query <http://help.brain-map.org/display/api/RESTful+Model+Access+%28RMA%29>`_.


