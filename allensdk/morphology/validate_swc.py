# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import allensdk.core.swc as swc
try:
    xrange
except:
    from past.builtins import xrange


def validate_swc(swc_file):
    """
    To be compatible with NEURON, SWC files must have the following properties:
        1) a single root node with parent ID '-1'
        2) sequentially increasing ID numbers
        3) immediate children of the soma cannot branch
    """
    soma_id = swc.Morphology.SOMA
    morphology = swc.read_swc(swc_file)
    # verify that there is a single root node
    num_soma_nodes = sum([(int(c['type']) == soma_id)
                          for c in morphology.compartment_list])
    if num_soma_nodes != 1:
        raise Exception(
            "SWC must have single soma compartment.  Found: %d" % num_soma_nodes)
    # sanity check
    root = morphology.root
    if root is None:
        raise Exception("Morphology has no root node")
    # verify that children of the root have max one child
    for root_child_id in root['children']:
        root_child = morphology.compartment_index[root_child_id]
        num_grand_children = len(root_child['children'])
        if num_grand_children > 1:
            raise Exception("Child of root (%s) has more than one child (%d)" % (
                root_child_id, num_grand_children))
    # get a list of all of the ids, make sure they are unique while we're at it
    all_ids = set()
    for compartment in morphology.compartment_list:
        iid = int(compartment["id"])
        if iid in all_ids:
            raise Exception("Compartment ID %s is not unique." %
                            compartment["id"])
        pid = int(compartment["parent"])
        if iid < pid:
            raise Exception(
                "Compartment (%d) has a smaller ID that its parent (%d)" % (iid, pid))
        all_ids.add(iid)

    # sort the ids and make sure there are no gaps
    sorted_ids = sorted(all_ids)
    for i in xrange(1, len(sorted_ids)):
        if sorted_ids[i] - sorted_ids[i - 1] != 1:
            raise Exception("Compartment IDs are not sequential")
    return True


def main():
    try:
        parser = argparse.ArgumentParser(
            "validate an SWC file for use with NEURON")
        parser.add_argument('swc_file')
        args = parser.parse_args()
        validate_swc(args.swc_file)
    except Exception as e:
        print(str(e))
        exit(1)
if __name__ == "__main__":
    main()
