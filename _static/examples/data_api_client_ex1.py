from allensdk.api.api import Api

class GeneAcronymQuery(Api):
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
