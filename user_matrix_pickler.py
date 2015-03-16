import pickle

user_to_index = pickle.load(open('user_to_index.p', 'rb'))
user_followee_matrix = {}

with open('friends.txt', 'r') as handle:
  # For each follower,
  for user, line in enumerate(handle):
    if user % 10 == 0:
      print user

    # Mark followees (must be an indexed user).
    for followee in line.split():
      if followee in user_to_index:
        followee_idx = user_to_index[followee]

        if followee_idx not in user_followee_matrix:
          user_followee_matrix[followee_idx] = {user: 1}
        elif user not in user_followee_matrix[followee_idx]:
          user_followee_matrix[followee_idx][user] = 1
        else:
          user_followee_matrix[followee_idx][user] += 1

# We are considering all indexed users, so some rows may be empty (ie. a user
# may not be followed by any of our indexed users) and some columns may be
# empty (ie. a user may not be following any of our indexed users).
# Followee x Follower matrix
pickle.dump(user_followee_matrix, open('user_followee_matrix.p', 'wb'))