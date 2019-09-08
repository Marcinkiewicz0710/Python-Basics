"""
String Basics
"""
# String to int and vice versa
int(s)
str(n)

# Last element
s[-1]

# First element
s[0] or s[-len(s)]

# Inverse String
s[::-1]

# Substring
s[<begin>:<end>:step]

# Prefix or suffix, return Boolean
s.startswith(<string>, <begin>, <end>)
s.endswith(<string>, <begin>, <end>)

# Split the string given a separator, return a list
s.split(' ')

# Transform a string of numbers into a list of numbers
l = list(map(int, s.split(' ')))

# Sort string in alphabetic order (sorted(s) returns a list)
''.join(sorted(s))

# Make all the letters lowercase/uppercase
s.lower()
s.upper()

# Check if all the letters are lowercase/uppercase
s.islower()
s.isupper()

"""
Char Basics
"""
# ASCII to Letter or Number
# 48->0  97->a  65->A
chr(c)

# Letter or Number to ASCII
ord(c)

