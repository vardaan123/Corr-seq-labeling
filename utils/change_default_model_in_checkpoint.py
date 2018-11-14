""" change the default model in the checkpoint file to a user specified model

This is useful when you want to backtrack to a previous model when doing early stopping

Essentially changes the first line of the checkpoint file to a user specificed path

Sample checkpoint file:

model_checkpoint_path: "/home/vikasraykar/deep/models/exp_1/checkpoints/model-9"
all_model_checkpoint_paths: "/home/vikasraykar/deep/models/exp_1/checkpoints/model-5"
all_model_checkpoint_paths: "/home/vikasraykar/deep/models/exp_1/checkpoints/model-6"
all_model_checkpoint_paths: "/home/vikasraykar/deep/models/exp_1/checkpoints/model-7"
all_model_checkpoint_paths: "/home/vikasraykar/deep/models/exp_1/checkpoints/model-8"
all_model_checkpoint_paths: "/home/vikasraykar/deep/models/exp_1/checkpoints/model-9"

"""

__author__  = "Vikas Raykar"
__email__   = "viraykar@in.ibm.com"

__all__ = ["change_default_model_in_checkpoint"]

from tempfile import mkstemp
from shutil import move
from os import remove, close

def change_default_model_in_checkpoint(checkpoint_file_path,model_file_path):
    """
    :params:
        checkpoint_file_path : str
            the path to the checkpoint file
        model_file_path : str
            the path to the model which you want to set as the default model
    """
    #Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(checkpoint_file_path) as old_file:
            for line in old_file:
                if 'model_checkpoint_path:' in line:
                    new_file.write('model_checkpoint_path: "%s"\n'%(model_file_path))
                else:                
                    new_file.write(line)
    close(fh)
    #Remove original file
    remove(checkpoint_file_path)
    #Move new file
    move(abs_path,checkpoint_file_path)

if __name__ == '__main__':
    """ example usage
    """
    checkpoint_file_path = r'/home/vikasraykar/deep/models/exp_1/checkpoints/checkpoint'
    model_file_path = r'/home/vikasraykar/deep/models/exp_1/checkpoints/model-7'
    change_default_model_in_checkpoint(checkpoint_file_path,model_file_path)
