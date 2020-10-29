import glob

encoding_types = ['One_hot', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix',
                  'Meiler_parameters', 'Acthely_factors', 'PAM250', 'BLOSUM62', 'Miyazawa_energies',
                  'Micheletti_potentials', 'AESNN3', 'ANN4D', 'ProtVec']
new_method_name = []
min_len = 32
begin = int(min_len/2)
for encoding_type in encoding_types:
    for i in range(begin, min_len+1, 2):
        forward_methodname = "forward_"+str(min_len)+"_"+encoding_type
        backward_methodname = "backward_"+str(min_len)+"_"+encoding_type
        new_method_name.append(forward_methodname)
        new_method_name.append(backward_methodname)
print(len(new_method_name))