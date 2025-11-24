import math, os, sys
import numpy                as np
import matplotlib.pyplot    as plt
import tensorflow_datasets  as tfds
import h5py

from scipy                  import ndimage
from tensorflow             import data, is_tensor, random
from cl_experiment.parsing  import Kwarg_Parser
from cl_experiment.utils    import log


import sys

class Dataset(object):
    ''' Externalizes the data generation. '''

    def __init__(self, **kwargs):
        self.parser = Kwarg_Parser(**kwargs)
        self.dataset_name       = self.parser.add_argument('--dataset_name',      type=str,     default='MNIST',        help='tfds download string or name of a compressed pickle file.')
        self.dataset_dir        = self.parser.add_argument('--dataset_dir',       type=str,     required=True,          help='set the default directory to search for or create dataset files.')
        if os.path.isabs(self.dataset_dir) == False:
            log.error("--dataset_dir must be absolute path")
            sys.exit(0)
        self.dataset_load       = self.parser.add_argument('--dataset_load',      type=str,     default='tfds', choices=['tfds', 'from_npz', 'hdf5'], help='determine which API to use for loading a dataset file.')
        self.data_type          = self.parser.add_argument('--data_type ',        type=int,     default=32,     choices=[32, 64], help='data type for .np/.tf arrays/tensors (32, 64).')

        self.renormalize01      = self.parser.add_argument('--renormalize01',     type=str,     default='no',   choices=['no', 'yes'], help='renormalize data by dividing all channels by its min/max -> [0, 1].')
        self.renormalize11      = self.parser.add_argument('--renormalize11',     type=str,     default='no',   choices=['no', 'yes'], help='renormalize data [0, +1] -> [-1, +1].')
        self.np_shuffle         = self.parser.add_argument('--np_shuffle',        type=str,     default='no',   choices=['no', 'yes'], help='shuffle the data (pre-loading)')
        self.vis_batch          = self.parser.add_argument('--vis_batch',         type=str,     default='no',   choices=['no', 'yes'], help='visualize a batch of pre-processed data?')
        self.vis_path           = self.parser.add_argument('--vis_path',          type=str,     required=True)
        if os.path.isabs(self.vis_path) == False: log.error("--vis_path must be absolute!")
        if not os.path.exists(self.vis_path): os.makedirs(self.vis_path)

        self.test_split         = self.parser.add_argument('--test_split',        type=float,   default=.1,     help='define the test split for unsupervised data')
        self.num_tasks          = self.parser.add_argument('--num_tasks',         type=int,     default=1,      help='specify the number of total tasks, in case of unsupervised: num_tasks splits given dataset into equal proportions!')
        self.batch_size         = self.parser.add_argument('--batch_size',        type=int,     default=100,    help='set the training batch size.')
        self.test_batch_size    = self.parser.add_argument('--test_batch_size',   type=int,     default=self.batch_size, help='set the test batch size.')
        self.epochs             = self.parser.add_argument('--epochs',            type=int,     default=16,     required=False, help='Training epochs!')

        self.random_seed        = self.parser.add_argument('--random_seed',       type=int,     default=-1,     help='set NP/TF random seed.')
        if self.random_seed != -1:
            np.random.seed(self.random_seed)
            random.set_seed(self.random_seed)
            
            
            
        self.domain_il               = self.parser.add_argument('--domain_il',           type=str,   default='no', choices=['no', 'yes'],        help='')
        self.block_size_tr           = self.parser.add_argument('--block_size_tr',       type=int,     default=30,     help='')
        self.block_size_tst          = self.parser.add_argument('--block_size_tst',       type=int,     default=30,     help='')
        # format: <nr_classes> [<task> <class> <newclass>]+
        self.remap_classes           = self.parser.add_argument('--remap_classes',       type=int,     default=[],     help='') ;
        if len(self.remap_classes)%3 != 1: 
          log.error("remap_classes must have length divisible by 3 + 1") ; 
          sys.exit(0) ;
        self.remap_classes = [self.remap_classes[0]] + list(np.array(self.remap_classes[1:]).reshape(-1,3)) ;
        print("Remap classes list ", self.remap_classes) ;

        #-------------------------------- LOADING (TFDS/NPZ)
        def load_via_tfds(dt):
            ''' Loads data via the tfds API. '''
            (x_tr, y_tr), info_tr = tfds.load(
                self.dataset_name,
                split='train',
                batch_size=-1,
                as_supervised=True,
                with_info=True
            )

            (x_tst, y_tst), _ = tfds.load(
                self.dataset_name,
                split='test',
                batch_size=-1,
                as_supervised=True,
                with_info=True
            )

            h, w, c         = info_tr.features['image'].shape
            num_classes     = info_tr.features['label'].num_classes

            log.debug(
                f'loaded raw tfds dataset:\n' +
                f'\ttrain\tx\t{type(x_tr)}\t{x_tr.shape}\ty\t{type(y_tr)}\t{y_tr.shape}\n' +
                f'\ttest\tx\t{type(x_tr)}\t{x_tst.shape}\ty\t{type(y_tst)}\t{y_tst.shape}\n' +
                f'\tdata dim:\t[{h},{w},{c}]\tclasses:\t{num_classes}'
            )

            #-------------------------------- ONE-HOT ENCODING
            y_tr_np          = y_tr.numpy().astype("int64")
            y_tst_np         = y_tst.numpy().astype("int64")

            onehot_y_tr_raw   = np.zeros([y_tr.numpy().shape[0], num_classes], dtype=dt)
            onehot_y_tst_raw  = np.zeros([y_tst.numpy().shape[0], num_classes], dtype=dt)

            onehot_y_tr_raw[range(0, y_tr_np.shape[0]), y_tr_np] = 1
            onehot_y_tst_raw[range(0, y_tst_np.shape[0]), y_tst_np] = 1

            #-------------------------------- RESHAPE / DTYPE CONVERSION
            x_tr = x_tr.numpy().astype(dt).reshape(-1, h, w, c)
            x_tst = x_tst.numpy().astype(dt).reshape(-1, h, w, c)

            log.debug(
                f'converted dataset (np):\n' +
                f'\ttrain\tx\t{type(x_tr)}\t{x_tr.shape}\ty\t{type(onehot_y_tr_raw)}\t{onehot_y_tr_raw.shape}\n' +
                f'\ttest\tx\t{type(x_tr)}\t{x_tst.shape}\ty\t{type(onehot_y_tst_raw)}\t{onehot_y_tst_raw.shape}'
            )

            return x_tr, onehot_y_tr_raw, x_tst, onehot_y_tst_raw
            

        def load_from_npz(dt):
            ''' Load data from a pickled file. '''
            #FIXME: detection if sup./unsup. 
            d = np.load(open(os.path.join(self.dataset_dir, self.dataset_name),"rb"))

            if self.dataset_name.endswith("npz"): 
                tr_x, tst_x , tr_y , tst_y = d.values() # a,b,c,d
                log.debug(
                    f'loaded raw .npz data:\n' +
                    f'tr_x: {tr_x.shape} tr_y: {tr_y.shape}\n' + 
                    f'tst_x: {tst_x.shape} tst_y: {tst_y.shape}\n' +
                    f'train sample min/max:\t[{np.min(tr_x)} {np.max(tr_x)}]\n'
                    f'test sample min/max:\t[{np.min(tst_x)} {np.max(tst_x)}]'
                )

                return tr_x.astype(dt), tr_y.astype(dt), tst_x.astype(dt), tst_y.astype(dt)
            else:
                x = np.load(os.path.join(self.dataset_dir, self.dataset_name))
                tst_cut = x.shape[0] - int(x.shape[0]*self.test_split) # calc. train/test split
                log.debug(
                    f'loaded raw .npy data:\n' +
                    f'x: {x.shape} max: {x.max()}'
                )
                
                return x[:tst_cut], None, x[tst_cut:], None


        def load_from_hdf5(dt):
            ''' Load data from a .hdf5 file. '''
            if type(self.dataset_name) == list:
                tr_path, tst_path = self.dataset_name[0], self.dataset_name[1]
                tr_path = os.path.join(self.dataset_dir, tr_path)
                tst_path = os.path.join(self.dataset_dir, tst_path)
            else: 
                tr_path = os.path.join(self.dataset_dir, self.dataset_name)
                tst_path = None
            log.debug(f"loading .hdf5...\ntrain: {tr_path}\ntest: {tst_path}")

            tr_f = h5py.File(tr_path, "r")

            tr_x, tr_y  = tr_f.get('x')[:], tr_f.get('y')[:]
            tr_f.close()

            tst_f = h5py.File(tst_path, "r")
            tst_x, tst_y = tst_f.get('x')[:], tst_f.get('y')[:]
            tst_f.close()

            return tr_x, tr_y, tst_x, tst_y

        #-------------------------------- LOAD
        if self.data_type == 32: dt = np.float32
        else: dt = np.float64

        self.raw_tr_xs, self.raw_tr_ys, self.raw_tst_xs, self.raw_tst_ys = None, None, None, None

        if self.dataset_load == 'from_npz':
            self.raw_tr_xs, self.raw_tr_ys, self.raw_tst_xs, self.raw_tst_ys = load_from_npz(dt)
        elif self.dataset_load == 'tfds':
            self.raw_tr_xs, self.raw_tr_ys, self.raw_tst_xs, self.raw_tst_ys = load_via_tfds(dt)
        elif self.dataset_load == 'hdf5':
            load_from_hdf5(dt)

        if len(self.raw_tr_xs.shape) == 4:
            _, h, w, c = self.raw_tr_xs.shape
        else:
            _, h, w = self.raw_tr_xs.shape
            c = 1
            self.raw_tr_xs = self.raw_tr_xs.reshape(-1, h, w, 1)
            log.debug(
                f'raw data:\n' + 
                f'train sample min/max:\t[{np.min(self.raw_tst_xs)} {np.max(self.raw_tst_xs)}]'
            )     
            self.raw_tst_xs = self.raw_tst_xs.reshape(-1, h, w, 1)
            log.debug(f'\nraw test sample min/max:\t[{np.min(self.raw_tst_xs)} {np.max(self.raw_tst_xs)}]')

        num_classes = self.raw_tr_ys.shape[1]
        self.properties = {'num_of_channels': c, 'num_classes': num_classes, 'dimensions': [h, w]}

        #-------------------------------- NORMALIZE
        def renormalize11(x_tr, x_tst=None): 
            ''' Perform a fixed normalization of pixel values. '''
            lower, upper = 0, +1
            tst = None
            tr = (upper - lower) * np.divide(
                np.subtract(
                    x_tr, np.min(x_tr)),
                np.subtract(
                    np.max(x_tr), np.min(x_tr))
            ) + lower
            tst = (upper - lower) * np.divide(np.subtract(x_tst, np.min(x_tst)), np.subtract(np.max(x_tst), np.min(x_tst))) + lower
            return tr, tst
        
        if self.renormalize11 == 'yes': 
            self.raw_tr_xs, self.raw_tst_xs = renormalize11(self.raw_tr_xs, self.raw_tst_xs)

        if self.renormalize01 == 'yes':
            lo, hi = np.min(self.raw_tr_xs), np.max(self.raw_tr_xs)
            self.raw_tr_xs = np.divide(
                self.raw_tr_xs - lo,
                hi - lo
            )
            self.raw_tst_xs = np.divide(
                    self.raw_tst_xs - lo,
                    hi - lo )
   
        if self.raw_tst_xs.shape[0] == 0: min_xs = ''; max_xs = ''
        else: min_xs = np.min(self.raw_tst_xs); max_xs = np.max(self.raw_tst_xs)
    
        log.debug(
            f'normalized data:\n' +
            f'\ttrain sample min/max:\t[{np.min(self.raw_tr_xs)} {np.max(self.raw_tr_xs)}]\n' +
            f'\ttest sample min/max:\t[{min_xs} {max_xs}]'
        )
        
        #-------------------------------- OUT!
        self.properties['train_shape']  = self.raw_tr_xs.shape
        self.properties['test_shape']   = self.raw_tst_xs.shape
        self.scalar_labels_train        = self.raw_tr_ys.argmax(axis=1)
        self.scalar_labels_test         = self.raw_tst_ys.argmax(axis=1)
        self.indices_train              = np.arange(self.raw_tr_xs.shape[0])
        self.indices_test               = np.arange(self.raw_tst_xs.shape[0])

        """ # INFO: comment in if you want to visualize a mini-batch at a dataset level.
        if self.vis_batch == 'yes':
            rnd_indices = np.random.random_integers(0, self.raw_tr_xs.shape[0], self.batch_size)
            img_data = self.raw_tr_xs[rnd_indices]
            label_data = None
            label_data = self.raw_tr_ys[rnd_indices]
            visualize_data(img_data, label_data, int(math.sqrt(self.batch_size)), self.vis_path, 'raw_input')
        """

    def get_class_indices(self, classes):
        ''' Returns indices of the data for specific classes. ''' 
        int_class   = int(classes)
        mask_train  = (self.scalar_labels_train == int_class)
        mask_test   = (self.scalar_labels_test == int_class)

        return self.indices_train[mask_train], self.indices_test[mask_test]


    def get_block_indices(self, classes):
        int_class   = int(classes)
        x = (int_class)*self.block_size_tr
        y = (int_class+1)*self.block_size_tr
        
        x_tst = (int_class)*self.block_size_tst
        y_tst = (int_class+1)*self.block_size_tst
        
        return self.indices_train[x:y], self.indices_test[x_tst:y_tst]
        
        
        
        
    def get_dataset(self, class_list, task_info=None, **kwargs):
        ''' Return a train/test dataset containing the specified classes. ''' 
        
        batch_size      = self.batch_size
        test_batch_size = self.test_batch_size

        #-------------------------------- FILTER CLASSES
        indices_set_train   = []
        indices_set_test    = []
        for class_ in class_list:
            if self.domain_il == 'yes':
                indices_train, indices_test = self.get_block_indices(class_)
            else:
                indices_train, indices_test = self.get_class_indices(class_)
               
            indices_set_train   += [indices_train]
            indices_set_test    += [indices_test]

        all_indices_train       = np.concatenate(indices_set_train, axis=0)        
        all_indices_test        = np.concatenate(indices_set_test, axis=0)
              

        # perform drop_remainder by hand
        nr_indices_train = (all_indices_train.shape[0] // self.batch_size) * self.batch_size
        nr_indices_test  = (all_indices_test.shape[0] // self.test_batch_size) * self.test_batch_size
        


        all_indices_train           = all_indices_train[0:nr_indices_train]
        all_indices_test            = all_indices_test[0:nr_indices_test]
          
        if self.np_shuffle == 'yes':
            np.random.shuffle(all_indices_train)
            np.random.shuffle(all_indices_test)

        data_train      = self.raw_tr_xs[all_indices_train]
        data_test       = self.raw_tst_xs[all_indices_test]
        labels_train, labels_test = [], []
        
        def remap_labels(task_info, l, mapping_info):
          if task_info is None or mapping_info == []: return l ;
          scalar = np.argmax(l, axis=1) ;
          nr_classes = mapping_info[0] ;
          lcp = np.zeros([l.shape[0], mapping_info[0]]) ;
          for (task, src, dst) in mapping_info[1:]:
            if task != task_info and task_info != 0: continue ; # we process labels if task_info matches the specified task or if specified task is 0, means "*"
            print("Remapping ", src, "to", dst, "for task", task_info) ;
            mask = (scalar == src) ;
            lcp[mask] = np.eye(nr_classes)[dst] ;
          return lcp ; 

        labels_train    = remap_labels(task_info, self.raw_tr_ys[all_indices_train], self.remap_classes) ;
        labels_test     = remap_labels(task_info, self.raw_tst_ys[all_indices_test], self.remap_classes) ;

        #-------------------------------- PREPARE DATA FORMAT (NUMPY)
        ds_obj_train    = (data_train, labels_train)    # TRAIN
        ds_obj_test     = (data_test, labels_test)      # TEST

        return ds_obj_train, ds_obj_test, data_train.shape[0], data_test.shape[0]


    def get_iterator(self, classes=None, iter_type='training', **kwargs):
        batch_size = self.batch_size
        if not classes:
            classes = kwargs.get('classes', range(10))

        ds_obj_train, ds_obj_test, _, _ = self.get_dataset(classes=classes, batch_size=batch_size, epochs=1)

        if iter_type == 'training': return iter(ds_obj_train)
        if iter_type == 'testing':  return iter(ds_obj_test)
        raise Exception('invalid type (default=training or testing)')


def visualize_data(data, label=None, save_path=None, filename='input'):
    if save_path is None: return
    else: 
        if not os.path.exists(save_path): os.makedirs(save_path)
        save_path = f'{save_path}/{filename}.png'
    non_quad = False
    if data.shape[1] == 1:
        non_quad = True
        N = data.shape[0]
        M = int(math.sqrt(data.shape[3]))+1

    ax_size = int(math.sqrt(data.shape[0]))
    fig, axes = plt.subplots(ax_size, ax_size, figsize=(ax_size, ax_size))
    
    for i in range(ax_size**2):
        img_data = data[i]
        if non_quad == True:
            flat = np.ravel(img_data)
            padded_data = np.resize(flat, M*M)
            img_data = np.reshape(padded_data, (M,M))
            K = 8; L = 8
            MK = M // K
            ML = MK
            img_data = img_data[:MK*K, :ML*L].reshape(MK, K, ML, L).max(axis=(1, 3))
        ax = axes[i//ax_size, i%ax_size]
        if is_tensor(img_data):
            img_data = img_data.numpy()
        ax.imshow(img_data.squeeze(), interpolation='nearest', cmap='viridis')
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if label is not None:
            if is_tensor(label):
                label = label.numpy()
            ax.set_title(f'label: {label[i].argmax()}')
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.axis('off')
    plt.savefig(save_path, transparent=False, bbox_inches='tight')
    plt.close('all')
