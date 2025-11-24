"""
Light-weight class allowing CL experiments includign logging of results based on bash files that can be
parameter-varied programmatically.

Assumptions are:
- create_model returns a class that has the usual keras methods: fit, call, predict, evaluate, save_weights, load_weights
- need not BE a keras object

tasks run from 1 to num_tasks 

"""
import sys, os
import math
import datetime
import numpy as np
import pandas as pd

from importlib import import_module

from cl_experiment.utils            import log, change_loglevel, helper, gpu_test
from cl_experiment.data             import Dataset
from cl_experiment.parsing          import Kwarg_Parser, Command_Line_Parser
from cl_experiment.callback         import Manager as Callback_Manager
from cl_experiment.checkpointing    import Manager as Checkpoint_Manager


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment:
    ''' Defines the experimental pipeline, this is the base class to derive more specific experiments from. '''

    def init_parser(self, **kwargs):
        ''' Reads command line arguments and stores them in a Kwarg_Parser instance. Argparse is NOT involved!! 
            The kwargs come from the instantiation of the Experiment class, these are additional args 
            passed to the constructor, usually empty.
        '''
        self.parser         = Kwarg_Parser(verbose=True, **kwargs)
        # ------------------------------------------------ MODEL
        self.model_module        = self.parser.add_argument('--model_module',          type=str,   default='DCGMM',                            help='class name in model sub-directory to instantiate')
        self.model_type          = self.parser.add_argument('--model_type',          type=str,   default='DCGMM',                            help='class name in model sub-directory to instantiate')
        # ------------------------------------------------ W&B
        self.log_level          = self.parser.add_argument('--log_level',           type=str,   default='INFO', choices=['DEBUG', 'INFO'],  help='enable printing and saving')
        if self.log_level == 'DEBUG': self.verbosity = 2
        else: self.verbosity = 0
        # ------------------------------------------------ DATA
        self.results_dir        = self.parser.add_argument('--results_dir',         type=str,   default='',                                 help='set the default directory to save the experiment result files. Empty string (default) means no storing of results.')
        self.dataset_dir        = self.parser.add_argument('--dataset_dir',         type=str,   default='./datasets',                       help='set the default directory to search for dataset files')
        self.dataset_name       = self.parser.add_argument('--dataset_name',        type=str,   default='mnist',                            help='loads a dataset via tfds. If not present, a download attempt is made. ')
        # ------------------------------------------------ EXPERIMENT PIPELINE: TRAINING
        self.epochs             = self.parser.add_argument('--epochs',              type=float, default=999.,                                help='number of training epochs per task')
        self.batch_size         = self.parser.add_argument('--batch_size',          type=int,   default=100,                                help='size of mini-batches we feed from train dataSet.')
        # ------------------------------------------------ EXPERIMENT PIPELINE: TESTING
        self.full_eval          = self.parser.add_argument('--full_eval',           type=str,   default='no', choices=['no', 'yes'],        help='test on all sub-tasks for FWT/BWT/Forgetting tracking')
        self.DAll               = self.parser.add_argument('--DAll',                type=str,   default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],     help='for each task test all given classes')
        self.test_batch_size    = self.parser.add_argument('--test_batch_size',     type=int,   default=100,                                help='batch size for testing')
        self.single_class_test  = self.parser.add_argument('--single_class_test',   type=str,   default='no', choices=['no', 'yes'],        help='enable testing on all single classes from the dataset')
        # ------------------------------------------------ CHECKPOINT
        self.exp_id             = self.parser.add_argument('--exp_id',              type=str,   default='0',                                help='unique experiment id (for experiment evaluation)')
        self.exp_tags           = self.parser.add_argument('--exp_tags',            type=str,   default=["dcgmm", "replay"],                help='sets some tags describing the purpose of the experiment')
        self.load_task          = self.parser.add_argument('--load_task',           type=int,   default=0,                                  help='load checkpoint stored AFTER load_task (0 = do not load checkpoint)')
        self.save_All           = self.parser.add_argument('--save_All',            type=str,   default='no', choices=['no', 'yes'],        help='saves the model for each task (after last training iteration)')
        # ------------------------------------------------- TASKS
        self.num_tasks          = self.parser.add_argument('--num_tasks',           type=int,   default=1,                                  help='specify the number of total tasks, in case of unsupervised: num_tasks splits given dataset into equal proportions!')
        
        # process epochs 
        self.task_list = []
        self.epochs = [self.epochs] * self.num_tasks if type(self.epochs) == type(0.5) else self.epochs
        if len(self.epochs) < self.num_tasks:
            log.error("epochs array too short")
            sys.exit(0)

        # process results_dir
        if self.results_dir != '':
          if os.path.isabs(self.results_dir) == False:
            log.error("--results_dir must be absolute!", self.results_dir)
            sys.exit(0)

        for i in range(1, self.num_tasks+1):
            args = self.parser.add_argument(f'--T{i}' , post_process=Kwarg_Parser.make_list, type=int, default=None, help='classes for the specific task, T1, T2, ..., TX, e.g. "--T1 0 1 2 3 4 5 6 7 8 9"')
            self.task_list.append(args)


    def __init__(self, from_cmd_line = True, **kwargs):
        ''' Prepare the experimental pipeline. Derived class contructors must call this!'''
        if from_cmd_line == True:
          command_line_params = Command_Line_Parser().parse_args()
          kwargs.update(command_line_params) ;
        self.init_parser(**kwargs)
        self.flags = self.parser.get_all_parameters()
        change_loglevel(self.log_level)
        self.init_dataset()
        self.init_variables()
        self.init_callbacks(**self.flags)
        self.checkpoint_manager = Checkpoint_Manager(**self.flags)
        self.test_results = {}


    #-------------------------------------------- DATA PREPARATION
    def init_dataset(self):
        ''' Initialize & pre-load the train/test datasets. '''
        self.dataset        = Dataset(**self.flags)

        self.properties     = self.dataset.properties
        self.h, self.w      = self.properties.get('dimensions')  # extract image properties or use slice patch size
        self.c              = self.properties.get('num_of_channels')
        self.num_classes    = self.properties.get('num_classes')
        self.flags['h']     = self.h
        self.flags['w']     = self.w
        self.flags['c']     = self.c

        if self.DAll != []: _, self.test_D_ALL, self.samples_train_D_ALL, self.samples_test_D_ALL = self.dataset.get_dataset(class_list=self.DAll, task_info = 0)


    def init_variables(self):
        ''' Initialize some experimental variables. '''
        self.global_iteration_counter   = 0
        self.tasks                      = [None] + [k for k in self.task_list if k is not None]
        self.training_sets              = [None]                # stores task training datasets
        self.test_sets                  = [None]                # stores task test datasets


    def load_task_dataset(self, current_task=1, preload_past_data=False):
        ''' 
        Create the sub-task datasets and append them to the global training and testing lists.
            * Parameters:
                - current_task: 1-based int
                - preload_past_data: bool flag if loaded from past task
             * Return:
                - self.iterations_task_all (int): return overall iterations for current task, passed to model.fit()
        '''
        if self.load_task > 0 and preload_past_data:  # if a checkpoint is loaded, preload the already processed datasets for full evaluation:
            log.info(f'preloading dataset from T1 - T{current_task}...')
            for task in range(1, current_task):
              self.load_task_dataset(current_task=task, preload_past_data=False)

        training, testing, samples_train, samples_test  = self.dataset.get_dataset(class_list=self.tasks[current_task], task_info=current_task)

        self.training_sets      += [training]
        self.test_sets          += [testing]

        log.info(f'loaded ds for T{current_task}), contained classes/proportions: {self.tasks[current_task]}, train set size: {samples_train}, test set size: {samples_test}')



    def ds_iter(self, cls, iter_type='training'):
        ''' Returns an iterator object for train/test containing the instances of given classes. '''
        return self.dataset.get_iterator(classes=cls, type=iter_type)

    #-------------------------------------------- MODEL CREATION/LOADING/SAVING
    def init_callbacks(self, **kwargs):
        self.train_callbacks, self.eval_callbacks = Callback_Manager(**kwargs).get_callbacks()


    # overwrite this mdethod to instantiate own model, or load existing one usong module and class names
    def create_model(self):
        ''' Imports and instantiates a functional keras model. '''
        log.debug(f'instantiating model of type "{self.model_type}"')
        model_module    = import_module(f'{self.model_module}')
        model_class     = getattr(model_module, self.model_type)
        model           = model_class(**self.flags)
        return model


    def load_model(self):
        ''' Loads the model saved in the task BEFORE self.load_task if --load_task was specified, it creates a new model from the config and tries to loads the weights from a checkpoint file. '''
        if self.load_task > 0:
            try:
                self.model = self.create_model() ;                                    # create a fresh model
                self.checkpoint_manager.load_checkpoint(self.model, self.load_task) ; # load a model state (saved model weights)
                return self.load_task + 1 ;
            except Exception as ex:
                raise ex ;
        else:
            self.model = self.create_model()                # do not load
            return 1 ;



    #-------------------------------------------- TRAINING/TESTING
    def before_task(self, task, **kwargs): 
        pass


    def run_experiment(self):
        ''' Main training loop:
            * loads/saves the model weights
            * loads the task dataset
            * executes all the training tasks
            * executes the test steps after a task has been finished
        '''
        self.before_experiment()
        self.start_task = self.load_model()                                                   # load model or create new one
        if self.full_eval == 'yes':                                         # load ALL datasets for full eval
            for task in range(1, self.num_tasks+1):
                self.load_task_dataset(task, preload_past_data=False)
        log.debug(f'starting training @T{self.start_task}, total tasks: {self.num_tasks}....')
        for task in range(self.start_task, self.num_tasks+1):                   # loop over all tasks (T1,T2,...,TX)
            if self.full_eval == 'no':
                pre_load = False if task > self.start_task else True
                self.load_task_dataset(task, preload_past_data=pre_load)       # load dataset for current & previous tasks if a model is loaded
            #if self.start_task == 1: self._test(task)
            self.before_task(task)                                              # before task procedure
            self.train_on_task(task)                                            # train on TX
            self.test(task)                                                    # test on various classes
            self.after_task(task)                                               # after task procedure
            if self.save_All == 'yes': self.checkpoint_manager.save_checkpoint(self.model, current_task=task)       # if parameter is set, create a checkpoint after each task
        if self.save_All == 'no': self.checkpoint_manager.save_checkpoint(self.model, current_task=self.num_tasks-1)  # save after whole training otherwise
        self.store_results()
        self.after_experiment()


    def train_on_task(self, current_task):
        ''' This function runs several training steps for the current task. '''
        #log.info('{:20s}'.format(f' [START] training on task: {current_task} epoch: {t_e} ').center(64, '~'))
        train_data  = self.training_sets[current_task]  # get a TF dataset iterator OR numpy tuple for the current task (TX)
        epochs      = int(self.epochs[current_task-1])  # task indices are 1-based
        if epochs < 1: return

        xs = train_data[0]
        ys = train_data[1]
        # do not shuffle! If we shuffle, we do it ourselves, important for invariance-generating GMMs...
        train_history = self.model.fit(
            x=xs,y=ys,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=self.train_callbacks,
            verbose=self.verbosity, shuffle = False
        )
        #steps_per_epoch=(self.iterations_task_all // epochs),

    def after_task(self, current_task, **kwargs): pass


    def test_on_task(self, test_dataset, current_task, test_task):
        ret = self.model.evaluate(x=test_dataset[0],y=test_dataset[1],
                                batch_size=self.test_batch_size,
                                callbacks=self.eval_callbacks,
                                verbose=self.verbosity,
                                return_dict=True)
        return ret



    def test(self, current_task, current_step=None, full_eval=False):
        ''' 
        Test procedure:
            * 1 epoch on: a combination/merge of all previous, current and following tasks (definition of DAll)
            * 1 epoch on: all individual previous tasks (T_1, ..., T_x-1)
            * 1 epoch on: current task (T_x)
        '''
        if full_eval == 'yes': task_end = self.num_tasks + 1
        else: task_end = current_task + 1
        

        log.info('{:13s}'.format(' [START] evaluation ').center(64, '~'))
        
        for test_task in range(0 if self.DAll != [] else 1, task_end):  # test: DAll, all previous, the current & (opt.) all future sub-tasks -> 0: DAll, 1: T1, 2: T2, ...
            test_dataset = self.test_D_ALL if test_task == 0 else self.test_sets[test_task]    # get dataset to test

            #print(np.argmax(list(test_dataset.as_numpy_iterator())[0][1], axis=1))

            task_name               = 'DAll' if test_task == 0 else f'T{test_task}'             # build task name
            self.model.test_task    = task_name                                                 # test task identifier
            
            task_info   = '(test classes: {})'.format( ','.join(                            # get tested classes
                    map(str, self.DAll if test_task == 0 else self.tasks[test_task])))
            #test_steps      = self.get_task_iterations(test_task)[1]                            # get num of test steps (batches)

            log.info(f'\t[TEST] -> {task_name} {task_info}')

            ret = self.test_on_task(test_dataset, current_task, test_task)
            if ret is None: ret = {}
            if type(ret) is type({}):
                mod_ret = {f"afterT{current_task}_test_on_T{test_task}_"+key:value for key, value in ret.items() }
                self.test_results.update(mod_ret)
        log.info('{:11s}'.format(' [END] evaluation ').center(64, '~'))


    def store_results(self):
        # log results if an explicit abspath has been provided
        # if not: return and let the model handle the storing of results via, e.g., callbacks
        if self.results_dir == "": return

        data = np.array([v for k,v in self.test_results.items()]).reshape(1, -1)
        cols = [k for k,v in self.test_results.items()]
        #print(data.shape, data, cols)
        df = pd.DataFrame(columns=cols, data=data)

        data = {}
        longest = -1
        for col,val in self.test_results.items():
          if type(val) is type(1.0) or type(val)== type(1) or type(val) == np.float32:
            val = np.array([val])
          if type(val) is type(np.zeros([1])):
            val = val.ravel()
          if len(val) > longest:
            longest = len(val)
         
          #print("Processing", col, longest, len(val))
          data[col] = val

        broadcasted_data = {}
        newcols =  []
        for col,val in data.items():
          length = len(val)
          pholder = np.zeros([longest])
          pholder[0:length] = val
          newcol = col + "_" + str(length)
          broadcasted_data[newcol] = pholder
          newcols.append(newcol)

        df = pd.DataFrame(columns=newcols,data=broadcasted_data)

        if not os.path.exists(self.results_dir): os.mkdir(self.results_dir)
        df.to_csv(os.path.join(self.results_dir, self.exp_id+".csv"))


    def before_experiment(self): pass


    def after_experiment(self): pass


    def plot_dot(self):
        ''' Plot the model graph (architecture) via the pydot package & GraphViz (pip install pydot, pip install graphviz). '''
        from tensorflow import keras
        keras.utils.plot_model(self.model, to_file=f'{self.model.vis_path}/model_plot.png', show_shapes=True, show_layer_names=True, rankdir="TB", dpi=96)
