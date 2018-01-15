#!/usr/bin/python2.7
import tsksvm
# import pdb; pdb.set_trace()

# PREPROCESS AA SCALES

# min-max normalized from coepra-scales-2.csv
scales_file = "data/coepra-scales-2.csv"  # scales(=643)xaa(=20)
scales = tsksvm.preprocess_aa_scales(scales_file)
