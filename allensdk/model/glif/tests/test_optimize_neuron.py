import os, re, logging
import allensdk.core.json_utilities as ju

from optimize_neuron import optimize_neuron

DATA_CONFIG_PATTERN = "/data/mat/Corinne/GLIF_subset/data_config_files/%d_data_config.json"

MODEL_CONFIG_DIR = "/local1/stash/allensdk/allensdk/model/glif/test/model_config/"
MODEL_CONFIG_FILES = [ os.path.join(MODEL_CONFIG_DIR, "329552531_LIF_model_config.json"),                      
                       os.path.join(MODEL_CONFIG_DIR, "476106176_LIF_R_ASC_AT_PWL_model_config.json"),
                       os.path.join(MODEL_CONFIG_DIR, "329552531_LIF_ASC_model_config.json") ]
                       
OUT_DIR = "/local1/stash/allensdk/allensdk/model/glif/test/optimized_config/"

def test_optimize_neuron():
    p = re.compile("(\d+)_(.*)_model_config.json")

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for model_config_file in MODEL_CONFIG_FILES:
        logging.info("testing %s" % model_config_file)

        fname = os.path.basename(model_config_file)
        m = p.match(fname)

        sid, config = m.groups()
        data_config_file = DATA_CONFIG_PATTERN % int(sid)

        data_config = ju.read(data_config_file)
        nwb_file = data_config["filename"]
        sweep_list = data_config["sweeps"].values()

        model_config = ju.read(model_config_file)

        #DBG
        model_config['optimizer']['inner_iterations'] = 1
        model_config['optimizer']['outer_iterations'] = 1
        #DBG
        
        sweep_index = { s['sweep_number']:s for s in sweep_list }    

        optimizer, best_param, begin_param = optimize_neuron(model_config, sweep_index)

        out_file = os.path.join(OUT_DIR, "%s_%s_neuron_config.json" % (sid, config))
        ju.write(out_file, optimizer.experiment.neuron.to_dict())

        out_config_file = os.path.join(OUT_DIR, "%s_%s_optimized_model_config.json" % (sid, config))
        ju.write(out_config_file, {
                'optimizer': optimizer.to_dict(),
                'neuron': optimizer.experiment.neuron.to_dict()
                })


if __name__ == "__main__": test_optimize_neuron()
