#!/usr/bin/python
import sys
import os
import h5py
import aif_common
import h5d_convert

TESTING = True

module_error = False
err_file = None
ai_file = None

ERR_FILE_NAME = "ephys_module.err"

# remove old error file if it's around
try:
	os.remove(ERR_FILE_NAME)
except OSError:
	pass

def module_exit():
	global err_file, module_error, ai_file
	if ai_file is not None:
		ai_file.close()
	if module_error:
		err_file.close()
		sys.exit(1)
	sys.exit(0)

def log_error(msg):
	global err_file, ERR_FILE_NAME
	if err_file == None:
		print "Writing error file '%s'" % ERR_FILE_NAME
		err_file = open(ERR_FILE_NAME, "w")
	print msg
	err_file.write(msg + "\n")
	module_error = True

if len(sys.argv) != 3:
	if TESTING:
		injson = "in.json"
		outjson = "out.json"
	else:
		log_error("Usage: %s <input json> <output json>")
		module_exit()

h5file = "157436/157436.03.01.h5"
ai_name = "157436.03.01.ai"

try:
	ai_file = h5py.File(ai_name, "r")
except OSError:
	assert False	# bug in code -- prevent overwrite
	h5d_convert.convert_h5d_aif(h5file, ai_name)
	ai_file = h5py.File(aifile, "r")

assert ai_file is not None, "Unable to open '%s'" % ai_name

err = ephys_qc.run_qc(ai_file, injson, outjson)

if err != None:
	log_error(err)

module_exit()

