import tensorflow as tf
import numpy as np
import detect_and_align
import id_data
from scipy import misc
import cv2
import os

from flask import Flask, request

from main import find_matching_id, get_embedding_distance, load_model, get_model_filenames, print_id_dataset_table, test_run

app = Flask(__name__)


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

    id_dataset = id_data.get_id_data('/ids', pnet, rnet, onet, sess, embeddings, images_placeholder, phase_train_placeholder)
    print_id_dataset_table(id_dataset)

    #test_run(pnet, rnet, onet, sess, images_placeholder, phase_train_placeholder, embeddings, id_dataset, args.test_folder)



def get_faces(frame):
    face_patches, padded_bounding_boxes, landmarks = detect_and_align.align_image(frame, pnet, rnet, onet)
    if len(face_patches) > 0:
        face_patches = np.stack(face_patches)
        feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
        embs = sess.run(embeddings, feed_dict=feed_dict)

        matches = []
        for i in range(len(embs)):
            bb = padded_bounding_boxes[i]

            matching_id, dist = find_matching_id(id_dataset, embs[i, :])
            if matching_id:
                matches.append(matching_id)
            else:
                matches.append('Unkown')
        return ', '.join(matches)
    else:
        return 'No faces'



@app.route('/identify', methods=['POST'])
def identify():
    data = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return get_faces(img)



if __name__ == "__main__":
    app.run(debug=True, port=8080, host='0.0.0.0')

