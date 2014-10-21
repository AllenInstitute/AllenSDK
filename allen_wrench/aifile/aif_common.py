
def parse_version(vers_str):
	""" Returns major, minor and patch numbers
	"""
	if not vers_string.startswith("AIFile"):
		return None, None, None
	v = vers_str.split('.')
	return v[1], v[2], v[3]

# returns date string based on number of seconds since midnight Jan 1st, 1904
def timestamp_string(sec):
	import datetime as dd
	d = dd.datetime.strptime("01-01-1904", "%d-%m-%Y")
	d += dd.timedelta(seconds=int(sec))
	return d.strftime("%a, %d %b %Y %H:%M:%S GMT")


def build_skeleton(hfile):
	import h5py
	hgen = hfile.create_group("general")
	hgen_ani = hgen.create_group("animal")
	hgen_dev = hgen.create_group("devices")
	hgen_ele = hgen.create_group("electrical")
	hgen_opt = hgen.create_group("optical")

	hacq = hfile.create_group("acquisition")
	hacq_seq = hacq.create_group("sequences")
	hacq_img = hacq.create_group("images")

	hstim = hfile.create_group("stimulus")
	hstim_temp = hstim.create_group("templates")
	hstim_pres = hstim.create_group("presentation")

	hepo = hfile.create_group("epochs")

	hproc = hfile.create_group("processing")

	hana = hfile.create_group("analysis")


def list_templates(hdf_file):
	import h5py
	stims = hdf_file["stimulus"]["templates"]
	lst = []
	for k in stims.keys():
		lst.append(k)
	lst.sort()
	return stims, lst
