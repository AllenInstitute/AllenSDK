from configure_model import configure_model
import allensdk.core.json_utilities as ju
import numpy as np
import re, os, logging

MODEL_CONFIG_FILES = [    
        "/data/mat/Corinne/GLIF_subset/model_config_files/369697038Nr5a1-CrePOS/369697038Nr5a1-CrePOS_LIF_R_ASC_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/369697038Nr5a1-CrePOS/369697038Nr5a1-CrePOS_LIF_R_ASC_AT_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/369697038Nr5a1-CrePOS/369697038Nr5a1-CrePOS_LIF_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/369697038Nr5a1-CrePOS/369697038Nr5a1-CrePOS_LIF_R_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/369697038Nr5a1-CrePOS/369697038Nr5a1-CrePOS_LIF_ASC_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/485912047Cux2-CreERT2POS/485912047Cux2-CreERT2POS_LIF_PWL_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/485912047Cux2-CreERT2POS/485912047Cux2-CreERT2POS_LIF_ASC_PWL_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/485912047Cux2-CreERT2POS/485912047Cux2-CreERT2POS_LIF_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/485912047Cux2-CreERT2POS/485912047Cux2-CreERT2POS_LIF_ASC_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/329552531Rorb-IRES2-CrePOS/329552531Rorb-IRES2-CrePOS_LIF_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/329552531Rorb-IRES2-CrePOS/329552531Rorb-IRES2-CrePOS_LIF_ASC_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/485455285Sst-IRES-CrePOS/485455285Sst-IRES-CrePOS_LIF_PWL_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/485455285Sst-IRES-CrePOS/485455285Sst-IRES-CrePOS_LIF_ASC_PWL_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/485455285Sst-IRES-CrePOS/485455285Sst-IRES-CrePOS_LIF_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/485455285Sst-IRES-CrePOS/485455285Sst-IRES-CrePOS_LIF_ASC_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_ASC_PWL_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_PWL_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_ASC_AT_PWL_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_ASC_AT_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_PWL_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_ASC_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_ASC_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_R_mlin_model_config.json",
        "/data/mat/Corinne/GLIF_subset/model_config_files/476106176Sst-IRES-CrePOS/476106176Sst-IRES-CrePOS_LIF_ASC_PWL_mlin_model_config.json"
]

def cmpdict(d1, d2):
    k1 = set([ unicode(k) for k in d1.keys() ])
    k2 = set([ unicode(k) for k in d2.keys() ])

    if len(k1 - k2) > 0:
        print sorted(list(k1))
        print sorted(list(k2))
        
        print d1
        print d2
        raise Exception("different keysets")

    for k in k1:
        v1 = d1[k]
        v2 = d2[k]

        try:
            if isinstance(v1, dict):
                cmpdict(v1, v2)
            elif isinstance(v1, list):
                if len(v1) != len(v2):
                    raise 

                for i in range(len(v1)):
                    if v1[i] != v2[i]:
                        raise
            elif d1[k] != d2[k]:
                raise
        except:
            raise Exception("%s: %s vs %s" % (k, str(d1[k]), str(d2[k])))
                            
def test_configure_model_old():
    p = re.compile("(\d+)(.*?)_(.*)_mlin_model_config.json")

    for mcf in MODEL_CONFIG_FILES:
        fname = os.path.basename(mcf)
        m = p.match(fname)
        
        sid, cre, config =  m.groups()

        method_config_file = "method_configurations/%s.json" % config
        prep_file = "/data/mat/Corinne/GLIF_subset/preprocessed_dicts/dictionaries/%s_preprocessed_dict.json" % sid

        neuron_config, optimizer_config = configure_model(ju.read(method_config_file), 
                                                          ju.read(prep_file))

        outd = ju.read(mcf)

        cmpdict(outd['neuron'], neuron_config)
        cmpdict(outd['optimizer'], optimizer_config)


def test_configure_model():
    p = re.compile("(\d+)(.*?)_(.*)_mlin_model_config.json")

    for mcf in MODEL_CONFIG_FILES:
        fname = os.path.basename(mcf)
        m = p.match(fname)
        
        sid, cre, config =  m.groups()

        method_config_file = "method_configurations/%s.json" % config
        prep_file = "test/%s_preprocessed_dict.json" % sid

        if os.path.exists(prep_file):
            print prep_file
            neuron_config, optimizer_config = configure_model(ju.read(method_config_file), 
                                                              ju.read(prep_file))
            
            outd = ju.read(mcf)
            
            cmpdict(outd['neuron'], neuron_config)
            cmpdict(outd['optimizer'], optimizer_config)

        else:
            logging.error("preprocessor file %s does not exist" % prep_file)


if __name__ == "__main__": test_configure_model()
    
    
