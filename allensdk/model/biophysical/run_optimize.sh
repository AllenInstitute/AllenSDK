set -o errexit

export ALLENSDK_PATH=/local1/git/allensdk

export NEURON_HOME=/shared/utils.x86_64/nrn-7.4-py27
export IV_HOME=/shared/utils.x86_64/iv-19
export PYTHONPATH=${NEURON_HOME}/lib/python:${ALLENSDK_PATH}:${PYTHONPATH}

export PYTHON_HOME=/shared/utils.x86_64/python-2.7
export PYTHON=${PYTHON_HOME}/bin/python
export PATH=${PYTHON_HOME}/bin:${NEURON_HOME}/x86_64/bin:${IV_HOME}/x86_64/bin:${PATH}

export SPECIMEN_ID=$1
export RUN_OPTIMIZE="${PYTHON} -W ignore -m allensdk.model.biophysical.run_optimize"
MANIFEST=manifest_sdk.json
OUT_JSON=out.json

${RUN_OPTIMIZE} generate_manifest_rma ${SPECIMEN_ID} ${MANIFEST}
${RUN_OPTIMIZE} copy_local ${MANIFEST} ${OUT_JSON}
${RUN_OPTIMIZE} nrnivmodl ${MANIFEST} ${OUT_JSON} 
${RUN_OPTIMIZE} start_specimen ${MANIFEST} ${OUT_JSON}
${RUN_OPTIMIZE} make_fit ${MANIFEST} ${OUT_JSON}
