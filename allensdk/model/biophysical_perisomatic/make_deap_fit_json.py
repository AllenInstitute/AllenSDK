import argparse
import os.path
import numpy as np
import json
from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.model.biophysical_perisomatic.fits.fit_styles
from pkg_resources import resource_filename #@UnresolvedImport
from allensdk.model.biophys_sim.config import DescriptionParser


class Report:

    def __init__(self,
                 description,
                 fit_type,
                 manifest_file_name='optimize_manifest_local.json'):
        self.description = description
        self.manifest = self.description.manifest
        self.specimen_id = str(description.data['runs'][0]['specimen_id'])
        self.fit_type = fit_type
        self.fit_directory = self.manifest.get_path('WORKDIR')
        self.target_path = self.manifest.get_path('target_path')
    
        with open(self.target_path, "r") as f:
            self.target = json.load(f)  # TODO: replace this w/ json utils
    
        self.seeds = [1234, 1001, 4321, 1024, 2048]
        self.org_selections = [0, 100, 500, 1000] # Picks thek best, 100th best, etc. organisms as examples
        self.trace_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
        
        self.config_path = self.manifest.get_path('fit_config_json',
                                                   self.fit_type)
         
        with open(self.config_path, "r") as f:
            self.fit_config = json.load(f)     # TODO: replace this w/ json utils
        
        fit_style_json = os.path.basename(self.fit_config["biophys"][0]["model_file"][-1])
        
        fit_style_path = \
            resource_filename(allensdk.model.biophysical_perisomatic.fits.fit_styles.__name__,
                              fit_style_json)
        
        with open(fit_style_path, "r") as f:
            self.fit_style_info = json.load(f)
        self.used_features = self.fit_style_info["features"]
        self.all_params = self.fit_style_info["channels"] + self.fit_style_info["addl_params"]
        
        nwb_path = self.manifest.get_path('output')
        self.data_set = NwbDataSet(nwb_path)
    
    
    def generate_fit_file(self):
        self.gather_from_seeds()
        self.make_fit_json_file()
    
    
    def make_fit_json_file(self):
        json_data = {}
        
        # passive
        json_data["passive"] = [{}]
        json_data["passive"][0]["ra"] = self.target["passive"][0]["ra"]
        json_data["passive"][0]["e_pas"] = self.target["passive"][0]["e_pas"]
        json_data["passive"][0]["cm"] = []
        for k in self.target["passive"][0]["cm"]:
            json_data["passive"][0]["cm"].append({"section": k, "cm": self.target["passive"][0]["cm"][k]})
        
        # fitting
        json_data["fitting"] = [{}]
        json_data["fitting"][0]["sweeps"] = self.target["fitting"][0]["sweeps"]
        json_data["fitting"][0]["junction_potential"] = self.target["fitting"][0]["junction_potential"]
        
        # conditions
        json_data["conditions"] = self.fit_style_info["conditions"]
        json_data["conditions"][0]["v_init"] = self.target["passive"][0]["e_pas"]
        
        # genome
        json_data["genome"] = []
        genome_vals = self.all_hof_fits[self.sorted_indexes[self.org_selections[0]], :]
        for i, p in enumerate(self.all_params):
            if len(p["mechanism"]) > 0:
                param_name = p["parameter"] + "_" + p["mechanism"]
            else:
                param_name = p["parameter"]
            json_data["genome"].append({"value": genome_vals[i],
                                        "section": p["section"],
                                        "name": param_name,
                                        "mechanism": p["mechanism"]
                                        })
        
        # write out file
        with open(self.manifest.get_path('output_fit_file',
                                         self.specimen_id,
                                         self.fit_type), "w") as f:
            json.dump(json_data, f, indent=2)


    def gather_from_seeds(self):
        first_created = False
        for s in self.seeds:
            final_hof_fit_path = self.manifest.get_path('final_hof_fit',
                                                        self.fit_type,
                                                        s)
            final_hof_path = self.manifest.get_path('final_hof',
                                                    self.fit_type,
                                                    s)
            if not os.path.exists(final_hof_fit_path):
                print "Could not find output file for seed {:d}".format(s)
                continue
            hof_fit_errors = np.loadtxt(final_hof_fit_path)
            hof_fits = np.loadtxt(final_hof_path)
            if not first_created:
                all_hof_fit_errors = hof_fit_errors.copy()
                all_hof_fits = hof_fits.copy()
                first_created = True
            else:
                all_hof_fit_errors = np.hstack([all_hof_fit_errors, hof_fit_errors])
                all_hof_fits = np.vstack([all_hof_fits, hof_fits])
        self.all_hof_fits = all_hof_fits
        self.all_hof_fit_errors = all_hof_fit_errors
        self.sorted_indexes = np.argsort(self.all_hof_fit_errors)


def main(description_path):
    fit_types = ["f6", "f9", "f12", "f13"]
    
    reader = DescriptionParser()
    description = reader.read(description_path)
    
    for fit_type in fit_types:
        fit_type_dir = description.manifest.get_path('fit_type_path', fit_type)
        
        if os.path.exists(fit_type_dir):
            report = Report(description,
                            fit_type)
            
            report.generate_fit_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate report on DEAP optimization run')
    parser.add_argument('manifest_json')
    args = parser.parse_args()
#    sns.set_style("white")

    main(args.manifest_json)


