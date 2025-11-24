import os, sys
import tensorflow as tf

from cl_experiment.utils    import log
from cl_experiment.parsing  import Kwarg_Parser


class Manager:
    '''  A manager supporting the saving/loading of training progress (model vars/weights) to the file system. '''

    def __init__(self, **kwargs):
        parser          = Kwarg_Parser(**kwargs)

        self.exp_id     = parser.add_argument('--exp_id', type=str, required=True)
        self.ckpt_dir   = parser.add_argument('--ckpt_dir', type=str, required=True, help='directory for checkpoint files')
        if os.path.isabs(self.ckpt_dir) == False:
            log.error("--chkpt_dir must be an absolute path!")
            sys.exit(0)
        if not os.path.exists(self.ckpt_dir): os.makedirs(self.ckpt_dir)
        self.filename   = os.path.join(self.ckpt_dir, f'{self.exp_id}-model_after_task_{{}}.weights.h5')


    def load_checkpoint(self, model, task , **kwargs):
        ''' Load a model configuration into the provided model  '''
 
        if task <= 0           : return ;

        ckpt_file = self.filename.format(task)
        try:
            model.load_weights(ckpt_file)
            log.info(f'restored model: {model.name} from checkpoint file "{ckpt_file}"...')
        except Exception as ex:
            log.error(f'a problem was encountered loading the model: {model.name} from checkpoint file "{ckpt_file}": {ex}')
            raise ex


    def save_checkpoint(self, model, current_task, **kwargs):
        ''' Saves the current session state to the file system. '''

        try:
            chkpt_filename = self.filename.format(current_task)
            model.save_weights(chkpt_filename)
            self.model_name = model.name
            log.info(f'saved model weights of "{self.model_name}" after task T{current_task} to file "{chkpt_filename}"')
        except Exception as ex:
            log.error(f'a problem was encountered saving the checkpoint file for model: {self.model_name} after task T{current_task}...')
            raise ex
