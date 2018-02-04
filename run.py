import tensorflow as tf
import numpy as np
import detect_and_align
import cv2
import pickle
import os
import re
import requests

from flask import Flask, request, jsonify


app = Flask(__name__)


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file



with tf.Graph().as_default():
    sess = tf.Session()
    #with tf.Session() as sess:

    pnet, rnet, onet = detect_and_align.create_mtcnn(sess, None)

    model_exp = '/model'
    print('Model directory: %s' % model_exp)
    meta_file, ckpt_file = get_model_filenames(model_exp)

    print('Metagraph file: %s' % meta_file)
    print('Checkpoint file: %s' % ckpt_file)

    saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
    saver.restore(sess, os.path.join(model_exp, ckpt_file))

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")



def get_embeddings(face_patches):
    face_patches = np.stack(face_patches)
    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
    return sess.run(embeddings, feed_dict=feed_dict).astype(np.float32)



@app.route('/identify', methods=['POST'])
def identify():
    data = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    face_patches, padded_bounding_boxes, landmarks = detect_and_align.align_image(img, pnet, rnet, onet)
    names = []
    distances = []
    if len(face_patches) > 0:
        embs = get_embeddings(face_patches)
        for emb in embs:
            data = {'vector': pickle.dumps(emb)}
            r = requests.post('http://localhost:5000/find', json=data)
            r_json = r.json()
            names.append(r_json.get('name', 'Unknown'))
            distances.append(r_json.get('distance', -1))
        return jsonify({'names': names, 'distances': distances})
    else:
        return 'No faces found'
    return jsonify({'vector': pickle.dumps(embs[0])})



@app.route('/add/<name>', methods=['POST'])
def add(name):
    data = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    face_patches, padded_bounding_boxes, landmarks = detect_and_align.align_image(img, pnet, rnet, onet)
    
    if len(face_patches) == 0:
        return 'No face was detected'
    elif len(face_patches) > 1:
        return 'For registering a name, please use pictures with only one face'

    embs = get_embeddings(face_patches)
    data = {'name': name,
            'vector': pickle.dumps(embs[0]),
            'model': 'tensorflow'
            }
    r = requests.post('http://localhost:5000/add', json=data)
    return r.text




if __name__ == "__main__":
    app.run(debug=True, port=8080, host='0.0.0.0')

