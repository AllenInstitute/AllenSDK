import allensdk.core.swc as swc

file_name = 'example.swc'
morphology = swc.read_swc(file_name)

# subsample the morphology 3x. root, soma, junctions, and the first child of the root are preserved.
sparse_morphology = morphology.sparsify(3)

# compartments in the order that they were specified in the file
compartment_list = sparse_morphology.compartment_list

# a dictionary of compartments indexed by compartment id
compartments_by_id = sparse_morphology.compartment_index

# the root compartment (usually the soma)
root = morphology.root

# all compartments are dictionaries of compartment properties
# compartments also keep track of ids of their children
for child_id in root['children']:
    child = compartments_by_id[child_id]
    print(child['x'], child['y'], child['z'], child['radius'])
