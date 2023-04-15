import h5py
import numpy as np
import os.path
from PIL import Image
from glob import glob
from skimage.transform import resize

raw_data_path = None
HDF5_data_path = './data/HDF5/'


def get_raw_data_path(dataset):
    if 'DRIVE' in dataset:
        raw_training_x_path = f'./data/{dataset}/training/images/*.tif'
        raw_training_y_path = f'./data/{dataset}/training/1st_manual/*.gif'
        raw_test_x_path = f'./data/{dataset}/test/images/*.tif'
        raw_test_y_path = f'./data/{dataset}/test/1st_manual/*.gif'
        raw_test_mask_path = f'./data/{dataset}/test/mask/*.gif'
    elif 'CHASEDB1' in dataset:
        raw_training_x_path = f'./data/{dataset}/training/images/*.jpg'
        raw_training_y_path = f'./data/{dataset}/training/1st_manual/*1stHO.png'
        raw_test_x_path = f'./data/{dataset}/test/images/*.jpg'
        raw_test_y_path = f'./data/{dataset}/test/1st_manual/*1stHO.png'
        raw_test_mask_path = f'./data/{dataset}/test/mask/*mask.png'
    elif 'STARE' in dataset:
        raw_training_x_path = f'./data/{dataset}/training/stare-images/*.ppm'
        raw_training_y_path = f'./data/{dataset}/training/labels-ah/*.ppm'
        raw_test_x_path = f'./data/{dataset}/test/stare-images/*.ppm'
        raw_test_y_path = f'./data/{dataset}/test/labels-ah/*.ppm'
        raw_test_mask_path = f'./data/{dataset}/test/mask/*mask.png'
    elif 'MMS' in dataset:
        raw_training_x_path = f'./data/{dataset}/training/images/*.bmp'
        raw_training_y_path = f'./data/{dataset}/training/manual/*.bmp'
        raw_test_x_path = f'./data/{dataset}/test/images/*.bmp'
        raw_test_y_path = f'./data/{dataset}/test/manual/*.bmp'
        raw_test_mask_path = f'./data/{dataset}/test/mask/*mask.bmp'
    else:
        return

    _raw_data_path = [raw_training_x_path, raw_training_y_path, raw_test_x_path,
                        raw_test_y_path, raw_test_mask_path]

    return _raw_data_path


def get_desired_data_shape(dataset):
    if 'DRIVE' in dataset:
        return (576, 576)
    elif 'CHASEDB1' in dataset:
        return (960, 960)
    elif 'STARE' in dataset:
        return (592, 592)
    elif 'MMS_P1_385_141_3' in dataset:
        return (385, 141)
    elif 'MMS_P2_385_141_3' in dataset:
        return (385, 141)
    elif 'MMS_P3_385_141_3' in dataset:
        return (385, 141)
    elif 'MMS_P1_385_385_3' in dataset:
        return (385, 385)
    elif 'MMS_P2_385_385_3' in dataset:
        return (385, 385)
    elif 'MMS_P3_385_385_3' in dataset:
        return (385, 385)

DESIRED_DATA_SHAPE = None


def isHDF5exists(raw_data_path, HDF5_data_path):
    for raw in raw_data_path:
        if not raw:
            continue

        raw_splited = raw.split('/')
        HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/*.hdf5'])

        if len(glob(HDF5)) == 0:
            return False

    return True


def read_input(path):
    if path.find('mask') > 0 and (path.find('CHASEDB1') > 0 or path.find('STARE') > 0):
        fn = lambda x: 1.0 if x > 0.5 else 0
        x = np.array(Image.open(path).convert('L').point(fn, mode='1')) / 1.
    elif path.find('2nd') > 0 and path.find('DRIVE') > 0:
        x = np.array(Image.open(path)) / 1.
    elif path.find('_manual') > 0 and path.find('CHASEDB1') > 0:
        x = np.array(Image.open(path)) / 1.
    else:
        x = np.array(Image.open(path)) / 255.
    if x.shape[-1] == 3:
        return x
    else:
        return x[..., np.newaxis]


def preprocessData(data_path, dataset):
    global DESIRED_DATA_SHAPE

    data_path = list(sorted(glob(data_path)))

    if data_path[0].find('mask') > 0:
        return np.array([read_input(image_path) for image_path in data_path])
    else:
        return np.array([resize(read_input(image_path), DESIRED_DATA_SHAPE) for image_path in data_path])


def createHDF5(data, HDF5_data_path):
    try:
        os.makedirs(HDF5_data_path, exist_ok=True)
    except:
        pass
    f = h5py.File(HDF5_data_path + 'data.hdf5', 'w')
    f.create_dataset('data', data=data)
    return


def prepareDataset(dataset):
    global raw_data_path, HDF5_data_path
    global DESIRED_DATA_SHAPE 
    
    DESIRED_DATA_SHAPE = get_desired_data_shape(dataset)
    raw_data_path = get_raw_data_path(dataset)

    if isHDF5exists(raw_data_path, HDF5_data_path):
        return

    for raw in raw_data_path:
        if not raw:
            continue

        raw_splited = raw.split('/')
        HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/'])

        preprocessed = preprocessData(raw, dataset)
        createHDF5(preprocessed, HDF5)


def getTrainingData(XorY, dataset):
    global HDF5_data_path

    raw_data_path = get_raw_data_path(dataset)
    raw_training_x_path, raw_training_y_path = raw_data_path[:2]

    if XorY == 0:
        raw_splited = raw_training_x_path.split('/')
    else:
        raw_splited = raw_training_y_path.split('/')

    data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
    f = h5py.File(data_path, 'r')
    data = f['data']

    return data


def getTestData(XorYorMask, dataset):
    global HDF5_data_path

    raw_test_x_path, raw_test_y_path, raw_test_mask_path = get_raw_data_path(dataset)[2:]

    if XorYorMask == 0:
        raw_splited = raw_test_x_path.split('/')
    elif XorYorMask == 1:
        raw_splited = raw_test_y_path.split('/')
    else:
        if not raw_test_mask_path:
            return None
        raw_splited = raw_test_mask_path.split('/')

    data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
    f = h5py.File(data_path, 'r')
    data = f['data']

    return data
