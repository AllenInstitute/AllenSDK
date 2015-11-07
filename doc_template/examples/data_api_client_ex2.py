from gene_acronym_query import GeneAcronymQuery

query = GeneAcronymQuery()
gene_info = query.get_data('ABAT')
for gene in gene_info:
    print("%s (%s)" % (gene['name'], gene['organism']['name']))
