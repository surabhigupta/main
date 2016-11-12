def context_windows(words, C=5):
    '''A generator that yields context tuples of words, length C.
       Don't worry about emitting cases where we get too close to
       one end or the other of the array.

       Your code should be quite short and of the form:
       for ...:
         yield the_next_window
    '''
    # START YOUR CODE HERE
    
    for i in xrange(len(words)-C+1):
        context_window = words[i:i+C]
        yield context_window
    
    # END YOUR CODE HERE


def cooccurrence_table(words, C=2):
    '''Generate cooccurrence table of words.
    Args:
       - words: a list of words
       - C: the # of words before and the number of words after
            to include when computing co-occurrence.
            Note: the total window size will therefore
            be 2 * C + 1.
    Returns:
       A list of tuples of (word, context_word, count).
       W1 occuring within the context of W2, d tokens away
       should contribute 1/d to the count of (W1, W2).
    '''
    table = []
    # START YOUR CODE HERE
    counts = dict()
    
    for context_window in context_windows(words, 2*C+1):
        for index, word in enumerate(context_window):
            if index == C:
                continue
            key = tuple([context_window[C], word])
            counts.setdefault(key, 0)
            d = abs(index - C)
            counts[key] += 1.0/float(d)
    
    for word_pair, count in counts.iteritems():
        table.append((word_pair[0], word_pair[1], float(count)))
    # END YOUR CODE HERE
    
    return sorted(table)


def score_bigram(bigram, unigram_counts, bigram_counts, delta):
    '''Return the score of bigram.
    See Section 4 of Word2Vec (see notebook for link).

    Args:
      - bigram: the bigram to score: ('w1', 'w2')
      - unigram_counts: a map from word => count
      - bigram_counts: a map from ('w1', 'w2') => count
      - delta: the adjustment factor
    '''
    # START YOUR CODE HERE
    if not (bigram[0] in unigram_counts and bigram[1] in unigram_counts):
        return 0.0
    return (float(bigram_counts[bigram] - delta)/(unigram_counts[bigram[0]] * unigram_counts[bigram[1]] ))
        
    # END YOUR CODE HERE
