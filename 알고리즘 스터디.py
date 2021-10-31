#!/usr/bin/env python
# coding: utf-8

# # 6장

# ## 문제1 팰린드롬

# In[25]:


string=input()


# In[26]:


string


# In[ ]:


'raceacar'


# In[27]:


def pelin(string):
    if len(string)==0:
        return False
    new_str=''
    for i in range(len(string)):
        if string[i].isalpha()==True:
            new_str+=string[i].lower()
    print(new_str)
    #if len(new_str)%2 ==0:
    counts=0
    entry=0
    ends=len(new_str)
    while counts<len(new_str)//2:
        if new_str[entry+counts] != new_str[ends-1-counts]:
            print(counts)
            print(new_str[entry+counts], '!!!',new_str[ends-1-counts])
            
            return False
            break
        counts+=1
    return True


# In[36]:


def isPalindrome(s:str) -> bool:
    strs =[]
    for char in s:
        if char.isalnum():
            strs.append(char.lower())
    
    while len(strs) >1:
        if strs.pop(0) != strs.pop():
            return False
    
    return True


# In[38]:


isPalindrome(s='aba')


# In[42]:


import collections


# In[46]:


string='abcba'


# In[ ]:


strs.pop(0)


# In[47]:


def isPalindrome(s:str)->bool:
    strs:Deque = collections.deque()
    
    for char in s :
        if char.isalnum():
            strs.append(char.lower())
    
    while len(strs) >1:
        if strs.popleft()!= strs.pop():
            return False
    
    return True


# In[48]:


isPalindrome(string)


# # deque가머냐?

# 정규표현식 -> 슬라이싱으로 한번에

# In[ ]:


왠만하면 슬라이싱이 빠르다


# In[ ]:





# # 2-문자열뒤집기

# In[66]:


list_=['h','e','l','l','o']


# In[61]:


list_[::-1]


# In[62]:


근데 이거 오류남 공간복잡도 떔에


# In[63]:


정답은 이거


# In[67]:


list_.reverse()
list_


# # 3- 로그재정렬

# In[2]:


logs=["dig1 8 1 5 1","let1 art can", "dig2 3 6", "let2 own kit dog","let3 art zero"]


# In[74]:


# for i in logs[0].split(' ')[1:]:
#     if i.isalpha()==False:
        
        
        


# In[6]:


def reorderLogFiles(logs):
    letters ,digits = [],[]
    for log in logs:
        if log.split()[1].isdigit():
            digits.append(log)
        else:
            letters.append(log)
    
    letters.sort(key=lambda x:(x.split()[1:],x.split()[0])) # 1기준으로 솔트하되 같으면 0으로
    return letters +digits


# In[7]:


reorderLogFiles(logs)


# In[9]:


'let1 art can'.split()[1:]


# In[ ]:





# # week 2

# # 4- 가장흔한단어

# In[1]:


paragraph= "Bob hit a ball, the hit BALL flew far after it was hit."


# In[2]:


banned=["hit"]


# In[4]:


lowering=paragraph.lower()


# In[8]:


lowering=lowering.replace(',','')
lowering=lowering.replace('.','')


# In[14]:


diction=lowering.split(' ')


# In[16]:


import collections


# In[17]:


counts=collections.defaultdict(int)


# In[20]:


for word in diction:
    counts[word]+=1


# In[21]:


counts


# In[23]:


for i in banned:
    counts.pop(i)


# In[28]:


counts.get(i)


# In[30]:


sorted(counts,key= counts.get)[-1]


# # 5 그룹 애너그램

# In[1]:


a=["eat","tea","tan","ate","nat","bat"]


# In[2]:


[["ate","eat","tea"],["nat","tan"],["bat"]]


# In[3]:


쏠팅을 하고 리스트에 추가방식? 근데이러면 다시돌려야함


# In[5]:


import collections


# In[6]:


anagrams=collections.defaultdict(list)


# 2개 생성할 것이 아니라 이런식으로 사전으루 추가해서 하나씩 늘려야함

# In[7]:


anagrams


# In[8]:


a


# In[ ]:





# In[41]:


for word in a:
    anagrams[''.join(sorted(word))].append(word)
    print(anagrams)


# # 6 가장 킨 팰린드롬 부분 문자열

# In[1]:


a="babadaaaavcxzzxcvaaa"


# In[2]:


pel=''


# In[ ]:


a


# In[3]:


for i in range(1,len(a)):
    for j in range(len(a)-i):
        if a[0+j:i+j] ==a[0+j:i+j][::-1]:
            pel=a[0+j:i+j]


# In[4]:


pel


# In[5]:


a[0:2][::-1]


# # 정답 팰린드롬 (2이상) 찾으면 확장하게 되있음

# In[76]:


def longestPalindrome(s):
    def expand(left,right):
        while left >=0 and right< len(s) and s[left]==s[right]:
            left-=1
            right+=1
        return s[left+1:right]
    
    if len(s)< 2 or s==s[::-1]:
        return s
    
    result = ''
    for i in range(len(s)-1):
        resutl = max(result,expand(i,i+1),expand(i,i+2),key=len)
    
    return result


# In[ ]:





# # 7 두수의 합

# In[9]:


nums = [2,7,11,15,3,4,102,34,29,10,5,6,79,7,8]
target=38
indexes=[]
for i in range(len(nums)-1,-1,-1):  
    if nums[i] <= target:
        indexes.append(i)
answer=''
for i in range(len(indexes)):
    for j in range(i+1,len(indexes)):
        if nums[indexes[i]]+ nums[indexes[j]] == target:
            answer=[indexes[j],indexes[i]]
if answer=='':
    print('답없당')
else:
    print(answer)


# In[ ]:





# # 답 1

# In[ ]:


def twosum(nums):
    for i , n in enumerate(nums):
        complement = target - n 
        
        if complement in nums[i+1:]:
            return [nums.index(n) , nums[i+1:].index(complement) + (i+1) ]


# # 답 2

# In[11]:


nums


# In[17]:


def twoSum(nums): 
    nums_map={}
    for i , num in enumerate(nums):
        nums_map[num] = i   # 2: 0 값이 키가 되고 키가 value 가 되어 들어감 
    
    for i , num in enumerate(nums):
        if target - num in nums_map and i != nums_map[target-num]: # 같은수 체크
            return [i,nums_map[target-num]]


# In[ ]:


딕셔너리 해쉬테이블이라 시간 O(1) 가능 


# In[19]:


target 9
1 != nums_map[2]


# In[18]:


twoSum(nums)


# ## 투포인타

# In[ ]:


def towSSum(nums):
    left, right =0, len(nums)-1
    while not left = right:
        if nums[left] + nums[right] < target:
            left +=1
        
        elif nums[left]+nums[right] > target:
            right -=1
        else:
            return [left,right]   # 답없을때 출력하는게 없음


# # 8 빗물 트래핑

# In[74]:


a= [0,1,0,2,1,0,1,3,2,1,2,1]


# 가로로 측정해서 빼는형식등로 해야할듯 

# In[75]:


height= list(set(a))[::-1]


# In[76]:


height


# In[88]:


a=[5,3,0,4,0,0,7]


# In[89]:


list(range(min(a),max(a)+1))


# In[ ]:





# In[11]:


a= [0,1,0,2,1,0,1,3,2,1,2,1]
height= list(range(min(a),max(a)+1))[::-1]


# In[12]:


height


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[90]:


a= [0,1,0,2,1,0,1,3,2,1,2,1]
#a=[5,3,0,4,0,0,7]
height= list(range(min(a),max(a)+1))[::-1]
water=0
for i in height:
    if a.count(i)==1:
        a[a.index(i)]=a[a.index(i)]-1
        continue
    start=a.index(i)
    end =len(a)-a[::-1].index(i)-1
    print(start,end)
    print(a)
    for j in range(start,end+1):
        if a[j]!=i:
            water+=1
        else:
            a[j]=a[j]-1
print(water)
            
    
            
            
                
            
            
        
        
        
    


# In[ ]:





# In[78]:


water


# In[26]:


a.count(1)


# # 9 세수의 합

# In[ ]:


from itertools import *


# In[51]:


import copy


# In[56]:


nums = [-1,0,1,2,-1,-4]


# In[8]:


indexing=list(range(len(nums)))


# In[9]:


indexing


# In[28]:


index_comb=list(combinations(indexing,2))
comb_values=list(combinations(nums,2))


# In[29]:


index_comb[0
          ]


# In[32]:


index_comb[0]


# In[30]:


pair


# In[37]:


pair[0]


# In[40]:


pair[0]


# In[43]:


for pair in zip(index_comb,comb_values):
    print(pair)


# In[61]:


answer=[]
for pair in zip(index_comb,comb_values):
    nums_copy=nums[::]
    nums_copy.pop(pair[0][1])
    nums_copy.pop(pair[0][0])
    if  (pair[1][0]+pair[1][1])*(-1) in nums_copy:
        answer.append([pair[1][0],pair[1][1],(pair[1][0]+pair[1][1])*(-1)])
        
        
        
        


# # 이거 시간복잡도 어케줄이지??

# In[65]:


answer


# # 10 배열파티션1

# In[67]:


nums=[1,4,3,2]


# In[79]:


sorting=sorted(nums)
sorting
sums=0
for x,y in enumerate(sorting):
    if x %2 ==0:
        sums+=y
print(sums)


# # 11 자신을 제외한 배열의 곱

# In[ ]:


nums=[1,2,3,4]


# for 문 밖에 생각아나는데? ㅡㅡ?

# # 12 주식
# 

# # 망함 ㅠㅠ

# In[107]:


#nums=[7,1,5,3,6,4]
nums = [7,6,4,3,1]
#nums=[4,1,2]
sorting=sorted(nums)


# In[108]:


nums


# In[109]:


sorting


# In[110]:


left=0
right=len(sorting)-1
while left <= right:
    diff = sorting[right] - sorting[left]
    print(left,right)
    if nums.index(sorting[right]) >= nums.index(sorting[left]):
        break
    if sorting[right]- sorting[right-1] > sorting[left+1]-sorting[left]:
        print('sadfsafaasdfdasfasdfsfsf')
        if nums.index(sorting[right]) >= nums.index(sorting[left+1]):
            diff = sorting[right] - sorting[left+1]
            break
        else:
            if nums.index(sorting[right-1]) >= nums.index(sorting[left]):
                diff = sorting[right-1] - sorting[left]
                break
    else:
        if nums.index(sorting[right-1]) >= nums.index(sorting[left]):
            diff = sorting[right-1] - sorting[left]
            break
        else:
            if nums.index(sorting[right]) >= nums.index(sorting[left+1]):
                diff = sorting[right] - sorting[left+1]
                break
    left-=1
    right+=1
    if left==right:
        diff=0
        
          
    


# In[1]:



import sys


# In[6]:


prices=[4,1,2]


# 

# In[2]:


sys.maxsize


# In[7]:


def maxProfit(prices):
    profit= 0
    min_price=sys.maxsize
    
    for price in prices:
        print(min_price,profit)
        min_price = min(min_price,price)
        profit = max(profit,price-min_price) # 시작은 0으로 시작됨. , 한번 끝까지 돌아야구나
    
    return profit


# In[ ]:


price min
0     100
0    4


# In[8]:


sys.maxsize


# In[9]:


maxProfit(prices)


# In[ ]:





# In[111]:


dif


# In[75]:


indexing


# In[49]:


nums.pop(-4)


# In[50]:


nums


# In[47]:


answer


# In[ ]:


fo


# In[25]:


pair


# In[2]:


from itertools import *


# In[7]:


list(combinations(nums,2))


# # 13 팰린드롬 연결리스트

# In[16]:


head=[1,2,3,4,2,4]


# In[17]:


def isPalindrome(head):
    if head==head[::-1]:
        return True
    else:
        return False
    
    


# In[18]:


head


# In[19]:


isPalindrome(head)


# In[20]:


head


# In[14]:


[1,2,2,4][::-1]


# In[15]:


head


# In[21]:


ListNode([12,3])


# In[22]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# In[38]:


ss=ListNode(1,3)


# In[41]:


ss.val


# In[37]:


ss.next


# In[36]:


list(ss)


# In[ ]:


def isPalindrome(self,head:ListNode) -> bool:
    q:List=[]
        
    if not head:
        return True
    
    node= head
    
    while node is not None:
        q.append(node.val)
        node=node.next
    
    while len(q) >1:
        if q.pop(0) != q.pop():
            return False

    return True


# # 같은 방식이지만 디큐로짜면 훨 빠르당

# In[ ]:


# deaue는 요렇게
q: Deque = collections.deque()


# # 러너 기법... 근데 이걸 알 필요가 있을까여...

# In[ ]:


def isPalindrome(self,ListNode) -> bool:
    rev=None
    slow=fast=head
    
    while fast and fast.next:
        fast=fast.next.next
        rev,rev.next, slow = slow, rev, slow.next
    if fast:
        slow=slow.next
    
    while rev and rev.val ==slow.val:
        slow , rev = slow.next ,rev.next
    return not rev


# # 14

# In[ ]:


def mergeTwoLists(self, l1:ListNode,l2:ListNode)-> ListNode:
    if (not l1) or (l2 and l1.val > l2.val):
        l1,l2 = l2, l1
    if l1:
        l1.next= sef.mergeTwoLists(l1.next , l2)
    
    return l1


# # 15

# In[ ]:


def reverseList(self,head:ListNode) -> ListNode:
    def reverse(node: ListNode , prev :ListNode=None):
        if not node:
            return prev
        next, node.next = node.next , prev
        return reverse(next ,node)
    return reverse(node)


# # 16

# In[ ]:


# 첫쨰풀이 짜집기방식 
# 역순 변환 후 , 리스트로 변환 , 덧셈하고 다시 연결리스트 변환


# In[8]:


root= head = ListNode(0)


# In[12]:


head.val


# In[ ]:


4 6 2
7 2 2


# In[ ]:


1 9 4


# In[6]:


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self,l1:ListNode,l2:ListNode):
        root= head = ListNode(0)
        carry = 0        # 값 저장 하는 곳임
        while l1 or l2 or carry:
            sum = 0
            if l1:
                sum +=l1.val
                l1=l1.next
            if l2:
                sum +=l2.val
                l2=l2.next
                                                    # 11 -> 1,1   / head에 0 , 1 담김
                                                    # divmod(8+1) -> 0,9  head에  0, 1, 9    4
            carry , val = divmod(sum+carry,10) 
            head.next = ListNode(val)
            head=head.next

        return root.next              # Root는 아직 0 이라  1, 9,4 나옴


# In[ ]:


[1,2,3,4,5]


# In[ ]:


12345


# In[1]:


import functools


# In[4]:


functools.reduce(lambda x,y: 10* x + y ,[1,2,3,4,5])


# In[13]:


1*10 + 2 = 12
12*10+3 =123
.
.
.
12345


# In[14]:


from sklearn.datasets import load_iris


# # 19

# In[ ]:


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if head is None:
            return None
        
        odd =head
        even=head.next
        even_head=head.next
      
        
        while even and even.next :
            odd.next , even.next = odd.next.next , even.next.next
            print(head,'1')
            odd , even = odd.next, even.next
            print(head,'2')
        odd.next = even_head
        return head


# # 스택

# # 20 유효한 괄호

# In[3]:


class Solution:
    def isValid(self, s: str) -> bool:
        s=s.replace("()",'')
        s=s.replace("[]",'')
        s=s.replace("{}",'')
        if s =='':
            return True
        else:
            return False


# In[ ]:


'[()]'


# In[ ]:


https://leetcode.com/problems/valid-parentheses/submissions/


# In[17]:


def isValid(s:str) -> bool:
    stack=[]
    table={
       ')' : '(',
       '}' : '{',
       ']' : '[',
    }
    
    for char in s :
        if char not in table:
            stack.append(char)
            print(stack)
        elif not stack or table[char] != stack.pop(): # 이러면 stack에서 하나 빠짐 
            print('stack')
            return False
    
    return (len(stack) ==0,'!')


# In[18]:


isValid(s='[][')


# In[ ]:





# In[4]:


g=[1,2,3]


# In[6]:


s=g.pop()


# In[7]:


s


# In[ ]:





# In[20]:


g=(len(x) for x in range(5))


# In[27]:


print(next(g))


# In[28]:


a=[1,2,3]
b=['a','b','c']


# In[29]:


for aa,bb in zip(a,b):
    print(aa, bb)


# # 21

# In[ ]:


ecbad


# In[16]:


string="ecbacdcbc"


# In[17]:


sorting=sorted(string)


# In[18]:


sorting


# In[20]:


set(sorting)


# In[21]:


string.index(sorting[0])


# In[24]:


direct=string[string.index(sorting[0]):]


# In[25]:


direct


# In[ ]:





# In[13]:


stack=[]


# In[29]:


list(range(1,2))


# In[31]:


stack


# In[ ]:


stack.sum()


# In[32]:


ggg=['a','b','c']


# In[34]:


gg=['a','d','b']


# In[39]:


gg=


# In[40]:


x


# In[37]:


gg.join('')


# In[30]:


for i in direct:
    if i not in stack:
        stack.append(i)
if len(stack)==len(set(sorting)):
    gg
    
    
    


# In[44]:


direct


# In[43]:


direct[::-1]


# In[ ]:


direct


# In[45]:


string.index(sorting[0])


# In[47]:


string


# In[ ]:


abce


# In[12]:


if len(direct) ==1:
    
    print(direct[0])
stack.append(direct[0])
for i in range(1,len(direct)):
    if direct[i]!=direct[i-1]:
        stack.append(direct[i])
        
if len(stack) == len(set(sorting)):
    answer=''
    for i in stack:
        answer+=i
else:
    stack=[]
    reverse=direct[0:string.index(sorting[0])][::-1]
    stack.append(reverse[0])
    for i in range(1,len(direct)):
        if direct[i]!=direct[i-1]:
            stack.append(direct[i])
    
        
    


# In[ ]:


sorting.replace()


# In[48]:


s=string


# In[49]:



s


# In[ ]:


for char in sored(set(s)):
    suffix = s[s.index(char):]
    if set(s) == set(suffix):
        return char + self.remo


# In[50]:


def remove(s):
    for char in sorted(set(s)):
        suffix = s[s.index(char):]
        if set(s) == set(suffix):
            return char + remove(suffix.replace(char,""))
    return ''


# In[51]:


remove('ecbacdcbc')


# In[64]:


'a'<'b'


# In[67]:


import collections
def remove1(s):
    counter, stack =collections.Counter(s),[]
    for char in s:
        counter[char] -=1
        if char in stack:
            continue
        while stack and char < stack[-1] and counter[stack[-1]] >0:
            stack.pop()
        stack.append(char)
        print(stack)
    return ''.join(stack)
    


# In[68]:


remove1('ecbacdcbc')


# In[61]:


remove1('ecbacdcbc')


# In[69]:


T=[73,74,75,71,69,72,76,73]


# In[70]:


[0]*7


# In[ ]:


class bcm()


# # 27

# In[4]:


lists=[[1,4,5],[1,3,4],[2,6]]


# In[5]:


remove=sum(lists,[])
# sorted(remove)


# In[6]:


remove


# In[7]:


sorted(remove)


# In[1]:


# 힙큐 


# In[ ]:


#https://littlefoxdiary.tistory.com/3


# In[ ]:


def mergeKLIsts(lists):
    root = result = ListNode(None)
    heap = []
    for i in range(len(lists)):
        if lists[i]:
            heapq.heappush(heap,(lists[i].val, i ,lists[i]))      # root 저장부분
            
    while heap:
        node = heapq.heappop(heap) # 가장작은 값이 나옴
        idx = node[1]
        result.next = node[2]
        
        result = result.next
        if result.next:
            heapq.heappush(heap, (result.next.val , idx , result.next))
        
    return root.next


# In[ ]:





# # 해쉬

# In[ ]:


# https://reakwon.tistory.com/128
# https://go-coding.tistory.com/30


# # 28

# In[3]:


1. 개별 체이닝 방식  
2. default dict 로 키없으면 ㅅㅐ성
3.  키 -> 해쉬함수 -> 인덱스


# # 29

# In[83]:


jewels="aA"
stones = "aAAbbbb"


# In[ ]:


3


# In[84]:


class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        after=stones         'bbbb'
        for i in jewels:
            after=after.replace(i,'')
            'aAAbbbb'  'bbbb'

        answer=(len(stones)-len(after))
        return answer


# In[87]:


a=Solution()


# In[88]:


a.numJewelsInStones(jewels,stones)


# # 문자열도되구나

# In[17]:


import collections
collections.Counter(S)


# In[18]:


sum(s in J for s in S) # s in j가 계산식 느낌.  s for s in S
[ (s in j) for s in S]


# # 30

# In[82]:


S="abcabcbb"
S='au'


# In[94]:


max_value=0
for s in range(len(S)):
    if len(S)==1:
        print(1)
    empty_dict=collections.defaultdict()
    for ss in range(s,len(S)):
        if ss==len(S)-1:
            max_value=len(empty_dict)
            return(max_value)
            break
        if S[ss] in empty_dict:
            if max_value < len(empty_dict):
                max_value=len(empty_dict)
            break
        else:
            empty_dict[S[ss]]=1
        


# In[90]:


max_value


# In[36]:


empty_dict=collections.defaultdict(int)


# In[42]:


empty_dict['b']='3'


# In[45]:


'a'  in empty_dict


# In[43]:


len(empty_dict)


# In[ ]:


'au' # 반례 ㅠㅠ


# In[ ]:


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s)==1:
            return 1
        max_value=0
        for ss in range(len(s)):
            empty_dict=collections.defaultdict()
            for sss in range(ss,len(s)):
                if s[sss] in empty_dict:
                    if max_value < len(empty_dict):
                        max_value=len(empty_dict)
                    break
                else:
                    empty_dict[s[sss]]=1
        return max_value


# In[82]:


enumerate 쓸걸 ㅠㅠ
[1,2,4]
[4,5,6]


# In[95]:


def lengthof(s):
    used={}
    max_length=start=0
    for index , char in enumerate(s):
        if char  in used and start <= used[char]:
            start = used[char]+1
        else:
            max_length = max(max_length , index -start +1)
        used[char]=index
    
    return max_length


# # 31

# In[1]:


nums= [1,1,1,2,2,3]
k=2


# In[100]:


dicts=collections.Counter(nums)
answer=[]
for i in dicts:
    if dicts[i] >=k:
        answer.append(i)
        


# In[6]:


import collections


# In[101]:


answer


# In[7]:


def topkfrequent(nums,k):
    return list(zip(*collections.Counter(nums).most_common(k)))[0]


# In[9]:


collections.Counter(nums).


# In[12]:


collections.Counter(nums).most_common(2)


# In[16]:


list(zip( collections.Counter(nums).most_common(2) ))


# In[17]:


list(zip( *collections.Counter(nums).most_common(2) ))


# In[8]:


topkfrequent(nums,k)


# In[ ]:


zip쓰면 두 리스트 튜플 형식으로  묶어줌
[1,2,3]
[4,5,6]
(1,4) , (2,5) , (3,6)


# In[ ]:


힙 뒤에서~


# In[ ]:


*은 (()) 이렇게 이중으로 들어가는거 풀어주는 역할임


# # 그래프 탐색

# ## dfs

# In[6]:


#깊이 우선 탐색이란


# In[1]:


graph={
    1:[2,3,4],
    2:[5],
    3:[5],
    4:[],
    5:[6,7],
    6:[],
    7:[3],
    
    
}


# # 재귀

# In[2]:


def recursive_dfs(v,discovered=[]):
    discovered.append(v)
    for w in graph[v]:
        if w not in discovered:
            discovered= recursive_dfs(w,discovered)
    return discovered


# In[3]:


recursive_dfs(1,[])


# # 반복

# In[4]:


def iterative_dfs(start_v):
    discovered=[]
    stack = [start_v]
    while stack:
        v=stack.pop()
        if v not in discovered:
            discovered.append(v)
            for w in graph[v]:
                stack.append(w)
    
    return discovered


# In[ ]:


성능차이 심하지 않다


# In[5]:


iterative_dfs(1)


# In[ ]:


https://www.youtube.com/watch?v=BLc3wzvycH8


# # BFS

# In[ ]:


https://www.youtube.com/watch?v=0v3293kcjTI


# In[7]:


def iterative_bfs(start_v):
    discovered= [start_v]
    queue = [start_v]
    while queue:
        v = queue.pop(0)
        for w in graph[v]:
            if w not in discovered:
                discovered.append(w)
                queue.append(w)
    return discovered


# In[ ]:





# In[8]:


iterative_bfs(1)


# In[ ]:


bfs 재귀 안된다 + 최소시간에서만 사용


# # 백트리킹은 뒤에서~

# In[75]:


class Cloth:
    def __init__(self,where):
        self.where = where
class Pants(Cloth):
    def __init__(self):
        self.price_range= '3만 ~ 5만'
Black_pants=Pants()
print(Black_pants.price_range)
print(Black_pants.where)


# In[89]:


a=sum([[1,2],[3,4,5]],[])
print(a)


# In[ ]:




