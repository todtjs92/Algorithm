import itertools
n = 2
data = ["N~F=0", "R~T>2"]
mem = ['A','C','F','J','M','N','R','T']
pairs = list(map(list,itertools.permutations(mem,8)))
count = 0

for check in data:
  first , second , case , num = check[0] , check[2] , check[3] , int(check[4])
  if case == '=':
    for pair in range(len(pairs)-1,-1,-1):
      n1 = pairs[pair].index(first)
      n2 = pairs[pair].index(second)
      if int(abs(n1-n2)) -1 != num:
        pairs.pop(pair)
  elif case == '>':
    for pair in range(len(pairs)-1,-1,-1):
      n1 = pairs[pair].index(first)
      n2 = pairs[pair].index(second)
      if int(abs(n1-n2)) -1 <= num:
        pairs.pop(pair)
  else:
    for pair in range(len(pairs)-1,-1,-1):
      n1 = pairs[pair].index(first)
      n2 = pairs[pair].index(second)
      if int(abs(n1-n2)) -1 >= num:
        pairs.pop(pair)
