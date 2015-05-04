Data API Client
===============


The `allensdk.api <allensdk.api.html>`_ package
is designed to help retrieve data from the Allen Brain Atlas Data Portal.
Currently there are only two API queries in Allen SDK.
The first is for the :py:class:`BiophysicalPerisomaticApi <allensdk.api.queries.biophysical_perisomatic_api.BiophysicalPerisomaticApi>` neuronal model.
The other is for :py:class:`GLIF <allensdk.api.queries.glif_api.GlifApi>` models.

Using an API Query
------------------

First import a query module:
    ::
    
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

This example will download a neuronal model to the working directory.
More information about running these models is available on the 
`perisomatic biophysical models <./biophysical_perisomatic_script.html>`_ page.



The API class
-------------

The query classes are derived from a generic
:py:class:`API class <allensdk.api.api>`.
It helps with composing a query URL and requesting the data.
Other methods read the response message and retrieve data files.

This simple subclass of Api provides methods to construct a query and access
the response json.  It then uses :py:meth:`~allensdk.api.api.Api.do_rma_query`
to execute the query.


gene_acronym_query.py:
    ::
    
        from allensdk.api.api import Api
        
        def GeneAcronymQuery(Api):
            def __init__(self):
                super(GeneAcronymQuery, self).__init__()
                
            def build_rma(self, acronym):
                return ''.join([self.rma_endpoint,
                               "/Gene/query.json",
                               "?criteria=",
                               "[acronym$il'%s']" % (acronym),
                               "&include=organism",
                               ])
            
            def read_json(self, json_parsed_data):
                if 'msg' in json_parsed_data:
                    return json_parsed_data['msg']
                else:
                    raise Exception("no message!")
            
            def get_data(self, acronym):
                return self.do_rma_query(self.build_rma,
                                         self.read_json,
                                         acronym)



The query class is then simple to use in a script.
main.py:
    ::
    
        from gene_acronym_query import GeneAcronymQuery
        
        query = GeneAcronymQuery()
        gene_info = query.get_data('ABAT')
        for gene in gene_info:
            print "%s (%s)" % (gene['name'], gene['organism']['name'])

Additional documentation is available to help with
`constructing an RMA query <http://help.brain-map.org/display/api/RESTful+Model+Access+%28RMA%29>`_.


