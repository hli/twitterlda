import pickle

counter = 0
user_to_index = {}
index_to_user = []

with open('users.txt', 'r') as handle:
  for index, user in enumerate(handle):
    user_id = user.strip()
    index_to_user.append(user_id)
    user_to_index[user_id] = index

pickle.dump(user_to_index, open('user_to_index.p', 'wb'))
pickle.dump(index_to_user, open('index_to_user.p', 'wb'))

print len(user_to_index)
print len(index_to_user)