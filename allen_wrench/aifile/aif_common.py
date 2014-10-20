
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


