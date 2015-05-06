Data API Client
===============


The `allensdk.api <allensdk.api.html>`_ package
is designed to help retrieve data from the
`Allen Brain Atlas API <http://help.brain-map.org/display/api/Allen+Brain+Atlas+API>`_.
Currently there are only two API queries in Allen SDK.
The first is for the :py:class:`Biophysical Perisomatic <allensdk.api.queries.biophysical_perisomatic_api.BiophysicalPerisomaticApi>` neuronal model.
The other is for :py:class:`GLIF <allensdk.api.queries.glif_api.GlifApi>` models.
The SDK can also be used to develop new queries.
 
 
Using an API Query
------------------

This example will download a biophysical perisomatic neuronal model to the working directory.
More information about running these models is available on the 
`perisomatic biophysical models <./biophysical_perisomatic_script.html>`_ page.
Another example of downloading and running GLIF models is available on the 
`GLIF models <glif_models.html#downloading-glif-models>`_ page.

First import a query module::

    from allensdk.api.queries.biophysical_perisomatic_api import \
        BiophysicalPerisomaticApi


For more information on using specific query modules, see the 
`queries package <allensdk.api.queries.html>`_ 
documentation for details.
::

    bp = BiophysicalPerisomaticApi()
    bp.cache_stimulus = False           # change to True to download the stimulus file
    bp.neuronal_model_id = 464137111    # get this from the web site
    bp.cache_data(neuronal_model_id, working_directory='neuronal_model')


Creating New API Query Classes
------------------------------

The following example demonstrates how to create a subclass of the generic
:py:class:`~allensdk.api.api.Api` class
in order to generate new queries and read data from the structure of
the response json.
The new class uses :py:meth:`~allensdk.api.api.Api.do_rma_query` to execute the query.

create gene_acronym_query.py::

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

The query class is then simple to use in a script.

create main.py::

    from gene_acronym_query import GeneAcronymQuery
    
    query = GeneAcronymQuery()
    gene_info = query.get_data('ABAT')
    for gene in gene_info:
        print "%s (%s)" % (gene['name'], gene['organism']['name'])

Additional documentation is available to help with
`constructing an RMA query <http://help.brain-map.org/display/api/RESTful+Model+Access+%28RMA%29>`_.


