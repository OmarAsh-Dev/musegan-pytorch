import gdown
from os.path import join

def download_dense_array(
    identifier,
    name,
    directory,
    ) -> None:
    output = join(directory,name)
    gdown.download(id = identifier, output = output, quiet = False)
    
