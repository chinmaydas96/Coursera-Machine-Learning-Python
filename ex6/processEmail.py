from string import lower
from porterStemmer import porterStemmer
from getVocabList import getVocabList
import re

def processEmail(email_contents):
    """preprocesses a the body of an email and
    returns a list of word_indices
    word_indices = PROCESSEMAIL(email_contents) preprocesses
    the body of an email and returns a list of indices of the
    words contained in the email.
    """

# Load Vocabulary
    vocabList = getVocabList()

# Init return value
    word_indices = []

# ========================== Preprocess Email ===========================

# Find the Headers ( \n\n and remove )
# Uncomment the following lines if you are working with raw emails with the
# full headers

# hdrstart = strfind(email_contents, ([chr(10) chr(10)]))
# email_contents = email_contents(hdrstart(1):end)

# Lower case
    email_contents = lower(email_contents)

# Strip all HTML
# Looks for any expression that starts with < and ends with > and replace
# and does not have any < or > in the tag it with a space
    rx = re.compile('<[^<>]+>|\n')
    email_contents = rx.sub(' ', email_contents)
# Handle Numbers
# Look for one or more characters between 0-9
    rx = re.compile('[0-9]+')
    email_contents = rx.sub('number ', email_contents)

# Handle URLS
# Look for strings starting with http:// or https://
    rx = re.compile('(http|https)://[^\s]*')
    email_contents = rx.sub('httpaddr ', email_contents)

# Handle Email Addresses
# Look for strings with @ in the middle
    rx = re.compile('[^\s]+@[^\s]+')
    email_contents = rx.sub('emailaddr ', email_contents)

# Handle $ sign
    rx = re.compile('[$]+')
    email_contents = rx.sub('dollar ', email_contents)

# ========================== Tokenize Email ===========================

# Output the email to screen as well
    print '==== Processed Email ====\n'

# Process file
    l = 0

# Remove any non alphanumeric characters
    rx = re.compile('[^a-zA-Z0-9 ]')
    email_contents = rx.sub('', email_contents).split()

    for str in email_contents:

        # Tokenize and also get rid of any punctuation
        # str = re.split('[' + re.escape(' @$/#.-:&*+=[]?!(){},''">_<#')
        #                                + chr(10) + chr(13) + ']', str)

        # Stem the word
        # (the porterStemmer sometimes has issues, so we use a try catch block)
        try:
            str = porterStemmer(str.strip())
        except:
            str = ''
            continue

        # Skip the word if it is too short
        if len(str) < 1:
           continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable str. You should look up str in the
        #               vocabulary list (vocabList). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if str = 'action', then you should
        #               look up the vocabulary list to find where in vocabList
        #               'action' appears. For example, if vocabList{18} =
        #               'action', then, you should add 18 to the word_indices
        #               vector (e.g., word_indices = [word_indices  18] ).
        #
        # Note: vocabList{idx} returns a the word with index idx in the
        #       vocabulary list.
        #
        # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
        #       str2). It will return 1 only if the two strings are equivalent.
        #




        # =============================================================

        # Print to screen, ensuring that the output lines are not too long
        if (l + len(str) + 1) > 78:
            print str
            l = 0
        else:
            print str,
            l = l + len(str) + 1

# Print footer
    print '\n========================='
    return word_indices

