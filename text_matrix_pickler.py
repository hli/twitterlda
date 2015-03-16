import pickle

word_to_index = pickle.load(open('word_to_index.p', 'rb'))
user_word_matrix = {}

with open('corpus.txt', 'r') as handle:
  for user, line in enumerate(handle):
    if user % 10 == 0:
      print user

    for word in line.split():
      word_idx = word_to_index[word]

      if user not in user_word_matrix:
        user_word_matrix[user] = {word_idx: 1}
      elif word_idx not in user_word_matrix[user]:
        user_word_matrix[user][word_idx] = 1
      else:
        user_word_matrix[user][word_idx] += 1

pickle.dump(user_word_matrix, open('user_word_matrix.p', 'wb'))