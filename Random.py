import random
import numpy as np

# randomly permute a list, the shuffled list will replace the previous one
random.shuffle(l)

# K samples from a given list without replacement
random.sample(l, K)
# Example: deal 20 cards without replacement
deck = collections.Counter(tens = 16, low_cards=36)
seen = random.sample(list(deck.elements()), 20)
print(seen.count('tens')/20)


# K samples from a given list with replacement
random.choice(l, K)
# Example: weighted sampling with replacement
random.choice(['red', 'black', 'green'], [18, 18, 2], 6)

# Generate random variables
# Uniform:
random.random()
random.uniform(a,b)

# Integer from 0 to n
random.randrange(n)
# Even integer from 0 to 100
random.randrange(0,101,2)

# Geometric distribution:
z = np.random.geometric(p=0.35, size=10000)
(z == 1).sum() / 10000. # The number of trials succeeded after a single run

# Binomial distribution:

# Poisson distribution:

# Exponential distribution:
