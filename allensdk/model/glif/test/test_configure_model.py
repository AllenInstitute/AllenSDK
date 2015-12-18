from configure_model import configure_model
import allensdk.core.json_utilities as ju
import numpy as np
import re, os, logging

MODEL_CONFIG_DIR = "/local1/stash/glif_clean/test/model_config_files/"

MODEL_CONFIG_FILES = [    
        os.path.join(MODEL_CONFIG_DIR, "329552531Rorb-IRES2-CrePOS/329552531Rorb-IRES2-CrePOS_LIF_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "329552531Rorb-IRES2-CrePOS/329552531Rorb-IRES2-CrePOS_LIF_ASC_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "369697038Nr5a1-CrePOS/369697038Nr5a1-CrePOS_LIF_R_ASC_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "369697038Nr5a1-CrePOS/369697038Nr5a1-CrePOS_LIF_R_ASC_AT_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "369697038Nr5a1-CrePOS/369697038Nr5a1-CrePOS_LIF_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "369697038Nr5a1-CrePOS/369697038Nr5a1-CrePOS_LIF_R_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "369697038Nr5a1-CrePOS/369697038Nr5a1-CrePOS_LIF_ASC_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "485912047Cux2-CreERT2POS/485912047Cux2-CreERT2POS_LIF_PWL_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "485912047Cux2-CreERT2POS/485912047Cux2-CreERT2POS_LIF_ASC_PWL_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "485912047Cux2-CreERT2POS/485912047Cux2-CreERT2POS_LIF_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "485912047Cux2-CreERT2POS/485912047Cux2-CreERT2POS_LIF_ASC_mlin_model_config.json"),

        os.path.join(MODEL_CONFIG_DIR, "485455285Sst-IRES-CrePOS/485455285Sst-IRES-CrePOS_LIF_PWL_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "485455285Sst-IRES-CrePOS/485455285Sst-IRES-CrePOS_LIF_ASC_PWL_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "485455285Sst-IRES-CrePOS/485455285Sst-IRES-CrePOS_LIF_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "485455285Sst-IRES-CrePOS/485455285Sst-IRES-CrePOS_LIF_ASC_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_ASC_PWL_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_PWL_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_ASC_AT_PWL_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_ASC_AT_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_PWL_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_ASC_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_ASC_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_mlin_model_config.json"),
        os.path.join(MODEL_CONFIG_DIR, "476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_ASC_PWL_mlin_model_config.json")
]

PREP_DIR = "/local1/stash/allensdk/allensdk/model/glif/test/preprocess/" 
OUT_DIR = "/local1/stash/allensdk/allensdk/model/glif/test/model_config/"
METHOD_CONFIG_DIR = "/local1/stash/allensdk/allensdk/model/glif/method_configurations/"

class ComparisonException( Exception ):
    def __init__(self, message, key):
        super(ComparisonException, self).__init__(message)
        self.key = key
        
def cmpdict(d1, d2):
    k1 = set([ unicode(k) for k in d1.keys() ])
    k2 = set([ unicode(k) for k in d2.keys() ])

    keys = k1 & k2
    disjoint_keys = k1 ^ k2
    
    if len(disjoint_keys) > 0:
        logging.warning("keys NOT in both dicts: %s" % str(disjoint_keys))
        
    for k in keys:
        v1 = d1[k]
        v2 = d2[k]

        try:
            if isinstance(v1, dict):
                cmpdict(v1, v2)
            elif isinstance(v1, list):
                if len(v1) != len(v2):
                    raise ComparisonException("", k)

                for i in range(len(v1)):
                    if v1[i] != v2[i]:
                        raise
            elif d1[k] != d2[k]:
                raise
        except:
            raise ComparisonException("%s: %s vs %s" % (k, str(d1[k]), str(d2[k])), k)
                            

def test_configure_model():
    p = re.compile("(\d+)(.*?)_(.*)_mlin_model_config.json")

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for mcf in MODEL_CONFIG_FILES:
        fname = os.path.basename(mcf)
        m = p.match(fname)
        
        sid, cre, config =  m.groups()

        method_config_file = os.path.join(METHOD_CONFIG_DIR, "%s.json" % config)
        prep_file = os.path.join(PREP_DIR, "%s_preprocessed_dict.json" % sid)
        out_file = os.path.join(OUT_DIR, "%s_%s_model_config.json" % (sid, config))

        if os.path.exists(prep_file):
            print "testing", prep_file
            out_config = configure_model(ju.read(method_config_file), 
                                         ju.read(prep_file))

            ju.write(out_file, out_config)

            test_config = ju.read(mcf)

            cmpdict(test_config['neuron'], out_config['neuron'])
            try:
                cmpdict(test_config['optimizer'], out_config['optimizer'])
            except ComparisonException, e:
                if e.key != "param_fit_names":
                    raise e
                

        else:
            logging.error("preprocessor file %s does not exist" % prep_file)


if __name__ == "__main__": test_configure_model()
    
    
