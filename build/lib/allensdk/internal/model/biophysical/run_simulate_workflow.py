import sys
from subprocess import call
from pkg_resources import resource_filename #@UnresolvedImport

the_script = resource_filename(__name__, 'run_simulate.sh')

cmd = ['/bin/bash', the_script]
cmd.extend(sys.argv[1:])

print(' '.join(cmd))

call(cmd)