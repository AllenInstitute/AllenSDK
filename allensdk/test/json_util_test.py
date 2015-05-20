import unittest
from mock import patch, mock_open
from allensdk.config.model.formats.json_util import JsonUtil

class JsonUtilTest(unittest.TestCase):
    commented_json = ("{\n"
                      "    // comment\n"
                      "    \"color\": \"blue\"\n"
                      "}")
    
    blank_line_json = ("{\n"
                      "\n"
                      "\n"
                      "\n"
                      "    \"color\": \"blue\"\n"
                      "}")
    
    multi_line_json = ("{\n"
                      "/* \n"
                      " * multiline comment\n"
                      " */\n"
                      "    \"color\": \"blue\"\n"
                      "}")
    
    two_multi_line_json = ("{\n"
                      "    \"colors\": [\"blue\",\n"
                      "    /* comment these out\n"
                      "    \"red\",\n"
                      "    \"yellow\",\n"
                      "     ... but not these */\n"
                      "    \"orange\",\n"
                      "    \"purple\",\n"
                      "    /* also comment this out\n"
                      "    \"indigo\",\n"
                      "    .... end comment */\n"
                      "    \"violet\"\n"
                      "    ]\n"
                      "}")
    
    
    def __init__(self, *args, **kwargs):
        super(JsonUtilTest, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        pass
    
    
    def tearDown(self):
        pass
    
    
    def testSingleLineComment(self):
        parsed_json = JsonUtil.read_json_string(
            JsonUtilTest.commented_json)
        
        self.assertTrue('color' in parsed_json and
                        parsed_json['color'] == 'blue')


    def testBlankLines(self):
        parsed_json = JsonUtil.read_json_string(
            JsonUtilTest.blank_line_json)
        
        self.assertTrue('color' in parsed_json and
                        parsed_json['color'] == 'blue')


    def testMultiLineComment(self):
        parsed_json = JsonUtil.read_json_string(
            JsonUtilTest.multi_line_json)
        
        self.assertTrue('color' in parsed_json and
                        parsed_json['color'] == 'blue')

    def testTwoMultiLineComments(self):
        parsed_json = JsonUtil.read_json_string(
            JsonUtilTest.two_multi_line_json)
        
        self.assertTrue('colors' in parsed_json)
        self.assertTrue(len(parsed_json['colors']) == 4)
        self.assertTrue('blue' in parsed_json['colors'])
        self.assertTrue('orange' in parsed_json['colors'])
        self.assertTrue('purple' in parsed_json['colors'])
        self.assertTrue('violet' in parsed_json['colors'])
    
    
    def testSingleLineCommentFile(self):
        with patch("__builtin__.open",
                   mock_open(
                       read_data=JsonUtilTest.commented_json)):
            parsed_json = JsonUtil.read_json_file('mock.json')
        
        self.assertTrue('color' in parsed_json and
                        parsed_json['color'] == 'blue')
    
    
    def testBlankLinesFile(self):
        with patch("__builtin__.open",
                   mock_open(
                       read_data=JsonUtilTest.blank_line_json)):
            parsed_json = JsonUtil.read_json_file('mock.json')
        
        self.assertTrue('color' in parsed_json and
                        parsed_json['color'] == 'blue')
    
    
    def testMultiLineFile(self):
        with patch("__builtin__.open",
                   mock_open(
                       read_data=JsonUtilTest.multi_line_json)):
            parsed_json = JsonUtil.read_json_file('mock.json')
        
        self.assertTrue('color' in parsed_json and
                        parsed_json['color'] == 'blue')
    
    
    def testTwoMultiLineFile(self):
        with patch("__builtin__.open",
                   mock_open(
                       read_data=JsonUtilTest.two_multi_line_json)):
            parsed_json = JsonUtil.read_json_file('mock.json')
        
        self.assertTrue('colors' in parsed_json)
        self.assertTrue(len(parsed_json['colors']) == 4)
        self.assertTrue('blue' in parsed_json['colors'])
        self.assertTrue('orange' in parsed_json['colors'])
        self.assertTrue('purple' in parsed_json['colors'])
        self.assertTrue('violet' in parsed_json['colors'])
