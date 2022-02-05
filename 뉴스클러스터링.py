import re
import itertools
import collections
def remover(lists):
  for i in range(len(lists)-1,-1,-1):
    if bool(re.search('[^a-z]',lists[i]))==True:
      lists.pop(i)
  return lists
def pairwise(string):
  empty =[]
  for i in range(0,len(string)-1):
    empty.append(string[i:i+2])
  return empty
def solution(str1, str2):
    str1= str1.lower()
    str2 = str2.lower()
    str1_list = pairwise(str1)
    str2_list = pairwise(str2)
    str1_list  = remover(str1_list)
    str2_list  = remover(str2_list)
    inter = list( (  collections.Counter(str1_list) & collections.Counter(str2_list)  ).elements() )
    outer = list( (  collections.Counter(str1_list) | collections.Counter(str2_list)  ).elements() )
    if len(outer) ==0:
        return 65536
    else:
        answer = int ( len(inter)/len(outer) * 65536 )
    return answer



# 이거 여쭤보기
inter = [str1_list.remove(x) for x in str2_list if x in str1_list]
