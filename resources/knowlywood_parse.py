"""
Each line in the file is of the form
verb; noun  predicate   num num
"""

file = open('knowlywood.txt', 'r')
print(len(list(file.readlines())))
for line in list(file.readlines())[:10]:
    #print(line.strip())
    verb_noun, predicate, num1, num2 = line.strip().split('\t')
    print(verb_noun.split(';'))

noun_gloss_file = open('noun.gloss', 'r')
print(len(list(noun_gloss_file.readlines())))
for line in list(noun_gloss_file.readlines())[1:10]:
    word, sense_number, synsetid, defi = line.strip().split('\t')
    print(word, ':', defi)
