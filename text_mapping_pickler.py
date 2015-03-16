import pickle

counter = 0
word_to_index = {}
index_to_word = []

with open('corpus.txt', 'r') as handle:
  for line in handle:
    for word in set(line.split()):
      if word not in word_to_index:
        index_to_word.append(word)
        word_to_index[word] = counter
        counter += 1

pickle.dump(word_to_index, open('word_to_index.p', 'wb'))
pickle.dump(index_to_word, open('index_to_word.p', 'wb'))
pickle.dump(len(word_to_index), open('vocab_size.p', 'wb'))

print len(word_to_index)
print len(index_to_word)