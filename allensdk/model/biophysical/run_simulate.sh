set -x
set -o errexit

export ALLENSDK_PATH=/local1/git/allensdk

export NEURON_HOME=/shared/utils.x86_64/nrn-7.4-py27
export IV_HOME=/shared/utils.x86_64/iv-19
export PYTHONPATH=${NEURON_HOME}/lib/python:${ALLENSDK_PATH}:${PYTHONPATH}

export PYTHON_HOME=/shared/utils.x86_64/python-2.7
export PYTHON=${PYTHON_HOME}/bin/python
export PATH=${PYTHON_HOME}/bin:${NEURON_HOME}/x86_64/bin:${IV_HOME}/x86_64/bin:${PATH}

export SPECIMEN_ID=$1
export SIMULATE="${PYTHON} -W ignore -m allensdk.model.biophysical.run_simulate"
MANIFEST=manifest_sdk.json
OUT_JSON=out.json

${RUN_SIMULATE} generate_manifest_rma ${SPECIMEN_ID} ${MANIFEST}
${RUN_SIMULATE} copy_local ${MANIFEST} ${OUT_JSON}
${RUN_SIMULATE} nrnivmodl ${MANIFEST} ${OUT_JSON} 
${RUN_SIMULATE} start_specimen ${MANIFEST} ${OUT_JSON}
