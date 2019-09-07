"""
Hash Table using collections package
"""
import collections
count = collections.Counter()

# To add/delete an element into the counter
count[x] += 1
count[x] -= 1

# Return a list of elements with respective repetition (iterable)
for item in count.elements():
	print(item)

# Return a list of (<item>, <frequency>) from the highest frequency to the lowest frequency
for item, freq in count.most_common():
	print(item + ' apprears ' + freq +' time(s).')

# If an element does not exist in the collection then count[x] will return 0


"""
Hash Table using Dictionary
"""
# Initialization and basic operations
dict = {}
dict['Lucie'] = 1
length = len(dict)
value = dict.get('Lucie')

# Return the value of the item and remove the item from the dictionary, return False if the item does not exist
value = dict.pop('Lucie', False)

# To iterate through the dictionary
for key, value in dict.iteritems():
	print(' The value of ' + key + ' is ' + value)

# To check if an item is in the dictionary
if item in dict:
	print('Bingo')


