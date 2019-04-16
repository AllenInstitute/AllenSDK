
IMAGE_SETS = {'Natural_Images_Lum_Matched_set_ophys_6_2017.07.14': '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_6_2017.07.14.pkl',
              'Natural_Images_Lum_Matched_set_training_2017.07.14': '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl',
              'Natural_Images_Lum_Matched_set_training_2017.07.14_2': '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl',
              'Natural_Images_Lum_Matched_set_ophys_6_2017.07.14_2': '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_6_2017.07.14.pkl'}



assert len(IMAGE_SETS) == len(set(IMAGE_SETS.keys())) == len(set(IMAGE_SETS.values()))