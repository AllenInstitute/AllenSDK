API Access
==========

The :py:mod:`allensdk.api` package is designed to help retrieve data from the
`Allen Brain Atlas API <http://help.brain-map.org/display/api/Allen+Brain+Atlas+API>`_. :py:mod:`~allensdk.api`
contains methods to help formulate API queries and parse the returned results.  There are several
pre-made subclasses available that provide pre-made queries specific to certain data sets. Currently there 
are several subclasses in Allen SDK:

    * :py:class:`~allensdk.api.queries.cell_types_api.CellTypesApi`: data related to the Allen Cell Types Database  
    * :py:class:`~allensdk.api.queries.biophysical_api.BiophysicalApi`: data related to biophysical models
    * :py:class:`~allensdk.api.queries.glif_api.GlifApi`: data related to GLIF models
    * :py:class:`~allensdk.api.queries.annotated_section_data_sets_api.AnnotatedSectionDataSetsApi`: search for experiments by intensity, density, pattern, and age
    * :py:class:`~allensdk.api.queries.grid_data_api.GridDataApi`: used to download 3-D expression grid data
    * :py:class:`~allensdk.api.queries.image_download_api.ImageDownloadApi`: download whole or partial two-dimensional images
    * :py:class:`~allensdk.api.queries.mouse_connectivity_api.MouseConnectivityApi`: common operations for accessing the Allen Mouse Brain Connectivity Atlas
    * :py:class:`~allensdk.api.queries.ontologies_api.OntologiesApi`: data about neuroanatomical regions of interest
    * :py:class:`~allensdk.api.queries.connected_services.ConnectedServices`: schema of Allen Institute Informatics Pipeline services available through the RmaApi
    * :py:class:`~allensdk.api.queries.rma_api.RmaApi`: general-purpose HTTP interface to the Allen Institute API data model and services
    * :py:class:`~allensdk.api.queries.svg_api.SvgApi`:  annotations associated with images as scalable vector graphics (SVG)
    * :py:class:`~allensdk.api.queries.synchronization_api.SynchronizationApi`: data about image alignment
    * :py:class:`~allensdk.api.queries.tree_search_api.TreeSearchApi`: list ancestors or descendents of structure and specimen trees 

RMA Database and Service API
----------------------------

One API subclass is the :py:class:`~allensdk.api.queries.rma_api.RmaApi` class.
It is intended to simplify
`constructing an RMA query <http://help.brain-map.org/display/api/RESTful+Model+Access+%28RMA%29>`_.

The RmaApi is a base class for much of the allensdk.api.queries
package, but it may be used directly to customize queries or to
build queries from scratch.

Often a query will simply request a table of data of one type:

::

    from allensdk.api.queries.rma_api import RmaApi
    
    rma = RmaApi()
    
    data = rma.model_query('Atlas',
                           criteria="[name$il'*Mouse*']")

This will construct the RMA query url, make the query and parse the resulting JSON
into an array of Python dicts with the names, ids and other information about the atlases
that can be accessed via the API.
                           
Using the criteria, include and other parameter, specific data can be requested.
::
    associations = ''.join(['[id$eq1]',
                            'structure_graph(ontology),',
                            'graphic_group_labels'])
    
    atlas_data = rma.model_query('Atlas',
                                 include=associations,
                                 criteria=associations,
                                 only=['atlases.id',
                                       'atlases.name',
                                       'atlases.image_type',
                                       'ontologies.id',
                                       'ontologies.name',
                                       'structure_graphs.id',
                                       'structure_graphs.name',
                                       'graphic_group_labels.id',
                                       'graphic_group_labels.name'])
                
Note that a 'class' name is used for the first parameter.
'Association' names are used to construct the include and criteria
parameters nested using parentheses and commas.
In the only clause, the 'table' form is used,
which is generally a plural lower-case version of the class name.
The only clause selects specific 'fields' to be returned.
The schema that includes the classes, fields, associations and tables
can be accessed in JSON form using:

::

    # http://api.brain-map.org/api/v2/data.json
    schema = rma.get_schema()
    for entry in schema:
        data_description = entry['DataDescription']
        clz = data_description.keys()[0]
        info = data_description.values()[0]
        fields = info['fields']
        associations = info['associations']
        table = info['table']
        print("class: %s" % (clz))
        print("fields: %s" % (','.join(f['name'] for f in fields)))
        print("associations: %s" % (','.join(a['name'] for a in associations)))
        print("table: %s\n" % (table))

Using Pandas to Process Query Results
-------------------------------------

When it is difficult to get data in exactly the required form
using only an RMA query, it may be helpful to perform additional
operations on the client side.  The pandas library can be useful
for this.

Data from the API can be read directly into a pandas 
`Dataframe <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_ object.

::

    import pandas as pd
    
    structures = pd.DataFrame(
        rma.model_query('Structure',
                        criteria='[graph_id$eq1]',
                        num_rows='all'))

`Indexing <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_
subsets of the data (certain columns, certain rows) is one use of pandas:
specifically `.loc <http://pandas.pydata.org/pandas-docs/stable/indexing.html#selection-by-label>`_:

::

    names_and_acromyms = structures.loc[:,['name', 'acronym']]

and `Boolean indexing <http://pandas.pydata.org/pandas-docs/stable/indexing.html#boolean-indexing>`_

::

    mea = structures[structures.acronym == 'MEA']
    mea_id = mea.iloc[0,:].id
    mea_children = structures[structures.parent_structure_id == mea_id]
    print(mea_children['name'])


`Concatenate, merge and join <http://pandas.pydata.org/pandas-docs/stable/merging.html>`_
are used to add columns or rows:

When an RMA call contains an include clause, the associated data will be represented
as a python dict in a single column.  The column may be converted
to a proper Dataframe and optionally dropped.

::

    criteria_string = "structure_sets[name$eq'Mouse Connectivity - Summary']"
    include_string = "ontology"
    summary_structures = \
        pd.DataFrame(
            rma.model_query('Structure',
                            criteria=criteria_string,
                            include=include_string,
                            num_rows='all'))
    ontologies = \
        pd.DataFrame(
            list(summary_structures.ontology)).drop_duplicates()
    flat_structures_dataframe = summary_structures.drop(['ontology'], axis=1)

Alternatively, it can be accessed using normal python dict and list operations.

::

    print(summary_structures.loc[:,'ontology'][0].name) 

Pandas Dataframes can be written to a CSV file using to_csv and read using load_csv.

::

    summary_structures[['id',
                        'parent_structure_id',
                        'acronym']].to_csv('summary_structures.csv',
                                           index_label='structure_id')
    reread = pd.DataFrame.from_csv('summary_structures.csv')


Iteration over a Dataframe of API data can be done in several ways.
The .itertuples method is one way to do it.

::

    for id, name, parent_structure_id in summary_structures[['name',
                                                             'parent_structure_id']].itertuples():
        print("%d %s %d" % (id, name, parent_structure_id))	


Caching Queries on Disk
-----------------------

:py:meth:`~allensdk.api.cache.Cache.wrap` has several parameters for querying the API,
saving the results as CSV or JSON and reading the results as a pandas dataframe.

::

    from allensdk.api.cache import Cache
    
    cache_writer = Cache()
    do_cache=True
    structures_from_api = \
        cache_writer.wrap(rma.model_query,
                          path='summary.csv',
                          cache=do_cache,
                          model='Structure',
                          criteria='[graph_id$eq1]',
                          num_rows='all')

If you change to_cache to False and run the code again it will read the data
from disk rather than executing the query.




