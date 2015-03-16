import numpy as np
import os
import pickle

FOLDER = 'topic'
index_to_word = pickle.load(open('index_to_word.p', 'rb'))
index_to_user = pickle.load(open('index_to_user.p', 'rb'))
convert_to_str = np.vectorize(lambda num: '%g' % num)
convert_to_word = np.vectorize(index_to_word.__getitem__)
convert_to_user = np.vectorize(index_to_user.__getitem__)

for filename in os.listdir(FOLDER):
  if filename.endswith('phi-top-words.npy'):
    item_filepath = os.path.join(FOLDER, filename)
    prob_filepath = os.path.join(FOLDER, filename.replace('words', 'phi'))
    top_items = np.load(open(item_filepath, 'rb'))
    top_prob = np.load(open(prob_filepath, 'rb'))

    output_filepath = os.path.join(FOLDER, filename.rstrip('.npy') + '.txt')
    np.savetxt(output_filepath,
               np.dstack((convert_to_word(np.fliplr(top_items).astype(int)),
                          convert_to_str(np.fliplr(top_prob)))),
               delimiter='\t',
               fmt = '%s')

  elif filename.endswith('sigma-top-users.npy'):
    item_filepath = os.path.join(FOLDER, filename)
    prob_filepath = os.path.join(FOLDER, filename.replace('users', 'sigma'))
    top_items = np.load(open(item_filepath, 'rb'))
    top_prob = np.load(open(prob_filepath, 'rb'))

    output_filepath = os.path.join(FOLDER, filename.rstrip('.npy') + '.txt')
    np.savetxt(output_filepath,
               np.dstack((convert_to_user(np.fliplr(top_items).astype(int)),
                          convert_to_str(np.fliplr(top_prob)))),
               delimiter='\t',
               fmt = '%s')