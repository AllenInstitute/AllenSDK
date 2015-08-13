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

For example, to download gene metadata from the API, first create gene_acronym_query.py:

.. literalinclude:: examples/data_api_client_ex1.py

The query class is then simple to use in a script.  Create main.py:

.. literalinclude:: examples/data_api_client_ex2.py

Additional documentation is available to help with
`constructing an RMA query <http://help.brain-map.org/display/api/RESTful+Model+Access+%28RMA%29>`_.


