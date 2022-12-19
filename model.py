from itertools import product 
import matplotlib.pyplot as plt
import numpy as np

# taken from model.wppl
model_webppl_dist = np.array([ 0.0559662090813094,
     0.3324450366422385,
     0.8921475875118259,
     0.702127659574468,
     0.953125,
     1.9542097488921713,
     1.743484224965706,
     0.6528925619834711,
     0.9323671497584539,
     1.9069767441860468,
     1.028397565922921,
     1.9806259314456034,
     1.547770700636943,
     1.8449502133712656,
     1.6075619295958283,
     0.38888888888888895 ])
model_webppl_dist /= sum(model_webppl_dist) # normalize

model_dist = np.array([.01, .02, .05, .06, .07, .15, .13, .04, .03, .09, .05, .06, .11, .12, .07, .02])
model_dist /= sum(model_dist) # normalize


def collapse_parity(s):
    if s == '': return
    
    if s[0] == 'H': d = {'H': '0', 'T': '1'}
    else: d = {'H': '1', 'T': '0'}
        
    return "".join([d[x] for x in s])

seq5s = ['0' + "".join(x) for x in product("01", repeat=4)]


def seq5_counts(seq):

    sub_seq5s = []
    for i in range(len(seq) - 4):
        j = i+4
        sub_seq5 = collapse_parity(seq[i:j+1])
        sub_seq5s.append(sub_seq5)

    def count(l, s):
        c = 0
        for x in l: 
            if x ==s: 
                c += 1
        return c

    counts = np.array([count(sub_seq5s, s) * 1.0 for s in seq5s])
#     dist = counts / np.linalg.norm(counts)
    
    return counts


def infer_probability(seq, model_dist):
    n = len(model_dist)
    
    counts = seq5_counts(seq)
    
    # prior on probability the input is human vs fair coin
    prior = .5
    
    model_likelihood = [np.log(model_dist[i] ** counts[i]) for i in range(n) if counts[i] != 0]      
    model_likelihood = np.sum(model_likelihood)
    
    fair_coin_likelihood = [np.log((1 / n) ** counts[i]) for i in range(n) if counts[i] != 0]
    fair_coin_likelihood = np.sum(fair_coin_likelihood)
    
#     print(model_likelihood, fair_coin_likelihood)
    posterior = np.exp(model_likelihood) * prior / (np.exp(model_likelihood) * prior + np.exp(fair_coin_likelihood) * (1-prior))
    #P(H | E) = P(E | H) P(H) / P(E)
    return posterior