import glob

file = "./output_data/saved_model/*.data-00000-of-00001"

files = glob.glob(file)
print(files)
