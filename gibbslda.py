"""
Adapted from implementation by Mathieu Blondel - 2010

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation for topics from two factors.
"""

import pickle
import numpy as np
import scipy as sp
from scipy.special import gammaln

def sample_index(p):
  """
  Sample from the Multinomial distribution and return the sample index.
  """
  return np.random.multinomial(1, p).argmax()

def word_indices(user_word_counts):
  """
  Turn a user word vector of size vocab_size to a sequence
  of word indices. The word indices are between 0 and
  vocab_size - 1. The sequence length is equal to the user text length.
  """
  for idx, count in user_word_counts.iteritems():
    for i in xrange(count):
      yield idx

def log_multi_beta(alpha, K=None):
  """
  Logarithm of the multinomial beta function.
  """
  if K is None:
    # alpha is assumed to be a vector
    return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
  else:
    # alpha is assumed to be a scalar
    return K * gammaln(alpha) - gammaln(K * alpha)

class LdaSampler(object):

  def __init__(self, n_topics, alpha=0.1, beta=0.1, gamma=0.1):
    """
    n_topics: desired number of topics
    alpha: a scalar (FIXME: accept vector of size n_topics)
    beta: a scalar (FIME: accept vector of size vocab_size)
    """
    self.n_topics = n_topics
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

  def _initialize(self, text_matrix, user_matrix):
    global vocab_size
    global n_users

    # number of words of user m labeled with topic z
    self.nmz = np.zeros((n_users, self.n_topics))
    # number of times topic z is labeled on word w
    self.nzw = np.zeros((self.n_topics, vocab_size))
    # number of followee links of user r labeled with topic z
    self.nrz = np.zeros((n_users, self.n_topics))
    # number of follower links of user m labeled with topic z
    self.nzm = np.zeros((self.n_topics, n_users))

    # total number of words of user m
    self.nm = np.zeros(n_users)
    # total number of times topic z is labeled on a word
    self.nz_word = np.zeros(self.n_topics)
    # total number of followee links of user r
    self.ne = np.zeros(n_users)
    # total number of follower links of user m
    self.nr = np.zeros(n_users)
    # total number of follow links with topic z
    self.nz_follow = np.zeros(self.n_topics)

    self.topics_for_words = {}
    self.topics_for_followers = {}

    for m in xrange(n_users):
      # i is a number between 0 and user_text_length - 1
      # w is a number between 0 and vocab_size - 1
      # text_matrix[m] is word counts for user m
      for i, w in enumerate(word_indices(text_matrix.get(m, {}))):
        # choose an arbitrary topic as first topic for word i
        z = np.random.randint(self.n_topics)
        self.nmz[m, z] += 1
        self.nm[m] += 1
        self.nzw[z, w] += 1
        self.nz_word[z] += 1
        self.topics_for_words[(m, i)] = z

      # follower is a number between 0 and n_users - 1
      # user_matrix[m] is follower bitmap for user m
      for i, follower in enumerate(user_matrix.get(m, {})):
        # choose an arbitrary topic as first topic for follower i
        z = np.random.randint(self.n_topics)

        # follower follows a user because of topic z
        self.nrz[follower, z] += 1
        # m is being followed by a user because of topic z
        self.nzm[z, m] += 1
        # a user is being followed by another user because of topic z
        self.nz_follow[z] += 1

        self.nr[m] += 1
        self.ne[follower] += 1
        self.topics_for_followers[(m, i)] = z

  def conditional_distribution_word(self, m, w):
    """
    Conditional distribution (vector of size n_topics) for word.
    """
    global vocab_size

    left = ((self.alpha + self.nzm[:, m] + self.nrz[m, :] + self.nmz[m, :]) /
            (self.alpha * self.n_topics + self.nr[m] + self.ne[m] + self.nm[m]))
    right = ((self.beta + self.nzw[:, w]) /
             (self.beta * vocab_size + self.nz_word))

    p_z = left * right
    # normalize to obtain probabilities
    p_z /= np.sum(p_z)
    return p_z

  def conditional_distribution_user(self, m, r):
    """
    Conditional distribution (vector of size n_topics) for user.
    """
    global vocab_size
    global n_users

    factor0 = self.alpha + self.nrz[r, :] + self.nzm[:, r] + self.nmz[r, :] + 1
    factor1 = factor0 - 1
    factor2 = self.alpha + self.nrz[m, :] + self.nzm[:, m] + self.nmz[m, :] + 1
    factor3 = factor2 - 1
    factor4 = ((self.gamma + self.nzm[:, r]) /
               (self.gamma * n_users + self.nz_follow))

    p_z = factor0 * factor1 * factor2 * factor3 * factor4
    # normalize to obtain probabilities
    p_z /= np.sum(p_z)
    return p_z

  def phi(self):
    """
    Compute phi = p(w|z).
    """
    num = self.nzw + self.beta
    num /= (self.beta * vocab_size + self.nz_word)[:, np.newaxis]
    return num

  def sigma(self):
    """
    Compute sigma = p(r|z).
    """
    num = self.nzm + self.gamma
    num /= (self.gamma * n_users + self.nz_follow)[:, np.newaxis]
    return num

  def run(self, text_matrix, user_matrix, maxiter=30):
    """
    Run the Gibbs sampler.
    """
    global n_users
    global vocab_size
    self._initialize(text_matrix, user_matrix)

    for it in xrange(maxiter):
      for m in xrange(n_users):
        for i, w in enumerate(word_indices(text_matrix.get(m, {}))):
          z = self.topics_for_words[(m, i)]
          self.nmz[m, z] -= 1
          self.nm[m] -= 1
          self.nzw[z, w] -= 1
          self.nz_word[z] -= 1

          p_z = self.conditional_distribution_word(m, w)
          z = sample_index(p_z)

          self.nmz[m, z] += 1
          self.nm[m] += 1
          self.nzw[z, w] += 1
          self.nz_word[z] += 1
          self.topics_for_words[(m, i)] = z

        for i, follower in enumerate(user_matrix.get(m, {})):
          z = self.topics_for_followers[(m, i)]
          self.nrz[follower, z] -= 1
          self.nzm[z, m] -= 1
          self.nz_follow[z] -= 1
          self.nr[m] -= 1
          self.ne[follower] -= 1

          p_z = self.conditional_distribution_user(m, follower)
          z = sample_index(p_z)

          self.nrz[follower, z] += 1
          self.nzm[z, m] += 1
          self.nz_follow[z] += 1
          self.nr[m] += 1
          self.ne[follower] += 1
          self.topics_for_followers[(m, i)] = z

      if it % RECORD_FREQUENCY == 0:
        yield (self.phi(), self.sigma())


if __name__ == "__main__":
  import os
  import shutil

  N_TOPICS = 100
  NUM_ITER = 100
  FOLDER = "topic"
  TOP_K = 40
  RECORD_FREQUENCY = 5

  if os.path.exists(FOLDER):
    shutil.rmtree(FOLDER)
  os.mkdir(FOLDER)

  text_matrix = pickle.load(open('user_word_matrix.p', 'rb'))
  user_matrix = pickle.load(open('user_followee_matrix.p', 'rb'))
  n_users = len(text_matrix)
  vocab_size = pickle.load(open('vocab_size.p', 'rb'))
  alpha = 50.0 / N_TOPICS
  beta = 200.0 / vocab_size
  gamma = 200.0 / n_users
  sampler = LdaSampler(N_TOPICS, alpha, beta, gamma)

  for it, (phi, sigma) in enumerate(sampler.run(text_matrix, user_matrix, NUM_ITER)):
    print "Iteration", it * RECORD_FREQUENCY

    top_sorted_items = np.empty((N_TOPICS, TOP_K))
    top_sorted_prob = np.empty((N_TOPICS, TOP_K))
    for z in xrange(N_TOPICS):
      topic_row = phi[z, :]
      top_items = np.argpartition(topic_row, -TOP_K)[-TOP_K:]
      sorted_top_idx = np.argsort(topic_row[top_items])
      top_sorted_items[z] = top_items[sorted_top_idx]
      top_sorted_prob[z] = topic_row[top_items][sorted_top_idx]

    np.save('topic/topic%.3d-phi-top-words.npy' % it, top_sorted_items)
    np.save('topic/topic%.3d-phi-top-phi.npy' % it, top_sorted_prob)

    top_sorted_items = np.empty((N_TOPICS, TOP_K))
    top_sorted_prob = np.empty((N_TOPICS, TOP_K))
    for z in xrange(N_TOPICS):
      topic_row = sigma[z, :]
      top_items = np.argpartition(topic_row, -TOP_K)[-TOP_K:]
      sorted_top_idx = np.argsort(topic_row[top_items])
      top_sorted_items[z] = top_items[sorted_top_idx]
      top_sorted_prob[z] = topic_row[top_items][sorted_top_idx]

    np.save('topic/topic%.3d-sigma-top-users.npy' % it, top_sorted_items)
    np.save('topic/topic%.3d-sigma-top-sigma.npy' % it, top_sorted_prob)