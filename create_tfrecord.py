# -*- coding: utf-8 -*-

"""
示例
python create_tfrecord.py --data_dir=data \
                          --output_dir=data \
                          --label_map_path=data/person_map.pbtxt
"""


from __future__ import absolute_import, division, print_function

import os
import io
import hashlib
import logging
import PIL
import tensorflow as tf

from lxml import etree
from object_detection.utils import label_map_util, dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', '')
flags.DEFINE_string('output_dir', '', '')
flags.DEFINE_string('label_map_path', '', '')

FLAGS = flags.FLAGS
SETS = ['train', 'val']


def dict_to_tf_example(data, images_dir, label_map_dict, ignore_difficult_instances=False):
    
    img_path = os.path.join(images_dir, data['filename'])
    with tf.gfile.GFile(img_path, 'rb') as f:
        encoded_jpg = f.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))
    
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
            }))
    return example
        

def main(_):
    data_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    images_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(images_dir):
        raise RuntimeError('`data_dir`目录下面需要有`images`图片目录')
    
    annotations_dir = os.path.join(data_dir, 'annotations')
    if not os.path.exists(annotations_dir):
        raise RuntimeError('`data_dir`目录下面需要有`annotations`标注xml目录')
    
    for s in SETS:
        
        writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, '{}.record'.format(s)))

        examples_path = os.path.join(data_dir, 'set', '{}.txt'.format(s))
        examples_list = dataset_util.read_examples_list(examples_path)
        
        for idx, example in enumerate(examples_list):
            path = os.path.join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data, images_dir, label_map_dict)
            writer.write(tf_example.SerializeToString())
        print('[INFO] `{}.record` 保存在 `{}` 成功'.format(s, output_dir))

        writer.close()
        

if __name__ == '__main__':
    tf.app.run()