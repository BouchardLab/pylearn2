import os, skimage.transform
from scipy.io import loadmat

def register_image(img, bbox, out_size):
    return skimage.transform.resize(img[bbox[0]:bbox[1],bbox[2]:bbox[3]], out_size)

def get_bbox(label_path):
    mat = loadmat(label_path)
    coords = mat['pts']
    return [coords[:,1].min(), coords[:,1].max(), coords[:,0].min(), coords[:,0].max()]

def get_label_attributes(label_path):
    label_dir, label_file = os.path.split(label_path)
    subject_id, session_num, recording_num, camera_label, image_num, _ = label_file.split('_')
    attrs = {}
    for k in ('subject_id', 'session_num', 'recording_num', 'camera_label', 'image_num'):
        attrs[k] = locals()[k]
    return attrs

def labelpath2impath(label_path):
    attrs = get_label_attributes(label_path)
    attrs['long_camera_label'] = attrs['camera_label'][:2] + '_' + attrs['camera_label'][2]
    image_path = 'Multi-Pie/data/session%(session_num)s/multiview/%(subject_id)s/%(recording_num)s/%(long_camera_label)s/%(subject_id)s_%(session_num)s_%(recording_num)s_%(camera_label)s_%(image_num)s.png' % attrs
    return image_path