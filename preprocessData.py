import os
import string
import gzip
import sys
import re

MIN_WORD_LENGTH = 3

#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#cachedStopWords = stopwords.words("english")

# Hardcode the stopwords from nltk to avoid module dependency
cachedStopWords = set(
  [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you',
   u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his',
   u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself',
   u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which',
   u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are',
   u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had',
   u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and',
   u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at',
   u'by', u'for', u'with', u'about', u'against', u'between', u'into',
   u'through', u'during', u'before', u'after', u'above', u'below', u'to',
   u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under',
   u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where',
   u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most',
   u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same',
   u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just',
   u'don', u'should', u'now'])

# http://www.netlingo.com/top50/popular-text-terms.php
slangStopWords = set(
  [u'2moro', u'2nite', u'brb', u'btw', u'b4n', u'bcnu', u'bff', u'cya',
   u'dbeyr', u'dilligas', u'fud', u'fwiw', u'gr8', u'ily', u'imho', u'irl',
   u'iso', u'jk', u'l8r', u'lmao', u'lol', u'lylas', u'mhoty', u'nimby', u'np',
   u'nub', u'oic', u'omg', u'ot', u'pov', u'rbtl', u'rotflmao', u'rt', u'thx',
   u'tx', u'thks', u'sh', u'sitd', u'sol', u'stby', u'swak', u'tfh', u'rtm',
   u'rtfm', u'tlc', u'tmi', u'ttyl', u'tyvm', u'vbg', u'weg', u'wtf', u'wywh',
   u'xoxo'])


try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET

# Gunzips and parses one XML Twitter user tweet file
# Returns one line with all of the tweets
def gzParseTweets(xmlPath):
  # reads gzip file
  xmlFile = gzip.open(xmlPath)
  fileContent = xmlFile.read()

  # Some XML files have an extra georss namespace
  # Removing the namespace if it exists
  modFileContent = string.replace(fileContent, 'georss:', '')

  # Converting contents into an XML Element Tree
  tweets = ""
  userIdx = None
  user = None
  root = ET.fromstring(modFileContent)

  for title in root.iter('title'):
    rawTweet = title.text
    if userIdx is None:
      userIdx = rawTweet.index(':')
      user = rawTweet[:userIdx].lower()
    tweet = cleanTweet(rawTweet[userIdx + 1:])
    tweets += " " + tweet
  return user, tweets

def cleanTweet(dirtyTweet):
  # Lowercase all of tweet
  dirtyTweet = dirtyTweet.lower()

  # Remove urls
  dirtyTweet = re.sub(r'(https?://|www\.)\S*', '', dirtyTweet)

  # Remove punctuation
  if type(dirtyTweet) is unicode:
    translate_table = dict((ord(char), u' ') for char in string.punctuation)
    dirtyTweet = dirtyTweet.translate(translate_table)
  else:
    replacePunctuation = string.maketrans(
      string.punctuation, ' ' * len(string.punctuation))
    dirtyTweet = dirtyTweet.translate(replacePunctuation)

  words = dirtyTweet.split()

  # Remove stopwords
  words = [word for word in words if (len(word) >= MIN_WORD_LENGTH and
                                      word not in cachedStopWords and
                                      word not in slangStopWords)]

  return ' '.join(words)

# Gunzips and parses one XML Twitter user friends file
# Returns one line with all of the friends
def gzParseFriends(xmlPath):
  # reads gzip file
  xmlFile = gzip.open(xmlPath)
  fileContent = xmlFile.read()

  # Some XML files have an extra georss namespace
  # Removing the namespace if it exists
  modFileContent = string.replace(fileContent, 'georss:', '')

  # Converting contents into an XML Element Tree
  friends = ""
  root = ET.fromstring(modFileContent)

  for id_elem in root:
    friend = id_elem.text
    friends += " " + friend
  return friends

# Collecting all of the friends and tweets in each run
corpus = []
users = []
userFriends = []
for root, dirs, files in os.walk('1'):
  foundTweets = False

  if 'tweets.rss.gz' in files:
    filePath = os.path.join(root, 'tweets.rss.gz')
    user_id = root.split('/')[-1]

    # reads gzipped xml files and extracts tweets
    user, tweets = gzParseTweets(filePath)
    user_id = root.split('/')[-1]
    if user is not None and tweets.strip() != '':
      users.append(user_id)
      corpus.append(tweets)
      foundTweets = True

  if foundTweets:
    if 'friends.rss.gz' in files:
      filePath = os.path.join(root, 'friends.rss.gz')
      friends = gzParseFriends(filePath)
      userFriends.append(friends)
    else:
      userFriends.append('')

# Writing the corpus into a file
corpusFile = open('corpus.txt', 'w')
corpusString = '\n'.join(corpus)

# Ran into some encoding issues, needed to encode everything into utf-8 before writing
corpusString = corpusString.encode('utf-8').strip()
corpusFile.write(corpusString)
corpusFile.close()

friendsString = '\n'.join(userFriends)
with open('friends.txt', 'w') as friendsFile:
  friendsFile.write(friendsString)

# Writing the users into a file
usersString = '\n'.join(users)
with open('users.txt', 'w') as usersFile:
  usersFile.write(usersString)