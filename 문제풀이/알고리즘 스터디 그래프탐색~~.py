#!/usr/bin/env python
# coding: utf-8

# In[ ]:


https://www.youtube.com/watch?v=BLc3wzvycH8


# # 32

# In[1]:


grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]


# In[7]:


len(grid[0][:])


# In[8]:


grid[:]


# In[10]:


for i in range(len(grid[:])):
    for j in range(len(grid[0][:])):
        if grid[i][j]=="1":
            grid[i][j]


# In[ ]:


def numislands(grid):
    def dfs(i,j):
        if i < 0 or i >= len(grid) or j < 0  or j >= len(grid[0]) or grid[i][j] != '1' :
            return
        
        grid[i][j] = 0
        
        dfs(i-1,j)
        dfs(i+1,j)
        dfs(i,j-1)
        dfs(i,j+1)
    
    count = 0
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] ==1:
                dfs(i,j)
                
                count+=1
    
    return count


# # 33

# In[2]:


import collections
import itertools


# In[3]:


number=dict()


# In[4]:


number['2']=['abc']
number['3']=['def']
number['4']=['ghi']
number['5']=['jkl']
number['6']=['mno']
number['7']=['pqrs']
number['8']=['tuv']
number['9']=['wxyz']


# In[6]:


def lettercombination(digits):
    def dfs(index , path):
        if len(path) ==len(digits):
            result.append(path)
            return
    
        for i in range(index,len(digits)):
            
            for j in dic[digits[i]]:
                dfs[i+1,path+j]
    
    if not digits:
        return []
    
    
    result = []
    dfs(0,"")
    
    return result


# In[5]:


number


# In[29]:


inputs="23"


# In[ ]:


for _ in range(len(inputs)):
    
    


# In[30]:


string=[]
for i in inputs:
    string+=number[i]


# In[31]:


string


# In[ ]:


itertools.


# In[33]:


list(itertools.combinations(string,2))


# # 34

# In[37]:


list(map(list,itertools.permutations([1,2,3])))


# # 35

# In[38]:


n=4
k=2


# In[39]:


a=list(range(1,n+1))


# In[41]:


list(map(list,itertools.combinations(a,k)))


# # 36

# In[3]:


import itertools


# In[2]:


candidates=[2,3,6,7]
target = 7


# In[7]:


list(itertools.combinations_with_replacement(candidates,2))


# In[5]:


itertools.combinations(candidates,2)


# In[5]:


def combinationSum(candidates,target):
    result=[]
    
    def dfs(csum,index,path): 
        if csum < 0:
            return
        
        if csum == 0:
            result.append(path)
            return
        
        for i in range(index, len(candidates)):                      # 2.  dfs(5, 0 , 2 )
            dfs(csum - candidates[i], i , path + [candidates[i]])
    
    dfs(target,0,[]) # 1. csum 에 타겟 , index 0 , path []
    
    return result
        


# In[ ]:





# In[6]:


combinationSum(candidates,target)


# # 37

# In[7]:


nums = [1,2,3]


# In[8]:


answer=[]
for i in range(len(nums)+1):
    answer+=list(map(list,itertools.combinations(nums,i)))
    
    


# In[9]:


answer


# # 38

# In[18]:


import collections


# In[17]:


start="JFK"


# In[24]:


del input


# In[69]:


input_list = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]


# In[80]:


input_list=[["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]


# In[77]:


travel_list=collections.defaultdict(list)


# In[74]:


travel_list=collections.defaultdict()


# In[75]:


travel_list


# In[73]:


travel_list


# ### 이거 첨알았네

# In[10]:


import collections


# In[11]:


start="JFK"


# In[12]:


input_list=[["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]


# In[13]:


travel_list=collections.defaultdict(list) # list로 값 담게 됨.


# In[14]:


for a,b in input_list:   # 경로 만들어주기
    travel_list[a].append(b)


# In[15]:


travel_list


# In[16]:


for i in travel_list:         # 정렬한번 해주고 이름순 땜에
    travel_list[i]=sorted(travel_list[i])


# In[17]:


answer=[]


# In[18]:


def dfs(start):                     # 스타트 들어오면 탐색 ㄱ ㄱ 
    answer.append(start)
    if travel_list[start]==[]:
        return
    print(start)  
    start=travel_list[start].pop(0)
    return dfs(start)
    
    
    
    

    
    
    


# In[19]:


dfs(start)


# In[184]:


answer


# # 40

# In[1]:


times=[[2,1,1],[2,3,1],[3,4,1]] 
N=4 
K=2


# In[17]:


times=[[2,1,1],[2,3,1],[3,4,1]] 
N=4 
K=2
import collections
import heapq # 힙큐 사용하는 이유는 힙팝으로 작은 값 뺄수있음 . 최단거리 탐색 .
# 네트워크시간 계산하는 문제라 탐색을 하되 가장 오래 걸린 시간만 반환하면 될듯.


def networkDelayTime(times , N , K ):
    graph = collections.defaultdict(list)
    
    for u , v , w in times :
        graph[u].append( ( v , w ) ) # graph 에는  출발지 u : (목적지 v , 시간 w) 가 담기게 됨. 지도
    #print(graph , 'graph')
    
    Q = [(0,K)] # K 출발지  # Q에는  (시간 ,목적지)  자기자신부터 시작 , 현재위치에서 탐색해야할 노드 경로들
    
    
    dist = collections.defaultdict(int) # dist 에 출발지부터 특정 노드까지의 시간이 담기게 됨 .
    
    while Q :  # while q 로 갈때 없을떄까지 탐색 .
        # print(Q ,'Q')
        
        time , node = heapq.heappop(Q) # 작은애 = 가장 가까운애 , 삭제하고 time , node 에 들어감.
        
        # print(time,node ,'time , node') # Q -> ( time , node)
        
        if node not in dist :   # dist에 없으면 = 아직 방문 안했으면 
            dist[node] = time            # dist -> {2: 0} = {node : time }
            for v, w in graph[node]:   # (v 목적지 , w 시간)
                alt = time + w          # alt에 시간 누적으로 적혀서 기록되고  푸시됨
                heapq.heappush(Q,(alt,v)) # 푸시로 넣어서 갈수 있는 곳 계속 탐색됨 .
        
        #print(dist , 'dist')
                            
    if len(dist) ==N :
        return max(dist.values())   # 가장 오래 걸린시간만 기록.

    return -1
    


# In[ ]:


o
    o   o
   o   o o o


# In[18]:


networkDelayTime(times , N , K )


# In[15]:


# 1까지 1거리 3까지 1 거리


# In[ ]:





# In[14]:


networkDelayTime(times , N , K )


# # 41

# In[2]:


import collections
import heapq


# In[4]:


o
    o   o
   o   o o o


# In[5]:


n=3 
flights= [[0,1,100], [1,2,100],[0,2,500]]
src = 0 # 시작점
dst = 2 # 목적지
K = 0 # 경유지 개수


# In[ ]:


o
    o   o
   o   o o or


# In[12]:


def findChepesPrice(n, flights , src , dst , K):
    graph = collections.defaultdict(list) # 경로 표시
    for u, v , w in flights:
        graph[u].append((v,w))
    print(graph ,'    graph')
    
    Q = [(0, src , K)] # 돈 , 출발지 , 경유지 개수   시작 은  0, 0, 2
  
    while Q:   # 현재위치에서 탐색해야할 노드 경로들
        print(Q,'                                        Q')
        price , node , k = heapq.heappop(Q)  # Q의  ( 돈 , 출발지 , 경유지 개수) 가 (price , node , k)
        if node ==dst:  # 출발지 자기자신일 경우 예외 처리 해줌
            return price
        
        if k>=0:
            for v, w in graph[node]: # v 목적지. , w 가 요금
                alt =price + w # alt 에 누적요금 
                heapq.heappush(Q,(alt, v, k-1)) #  # ( 누적요금 , 목적지가 다시 스타트로 , 한번 지났으니 k= k-1)
    return -1


# In[13]:


findChepesPrice(n, flights , src , dst , K)


# In[ ]:





# # 트리

# # 42 트리 깊이 재기

# In[1]:


tree = [3,9,20 ,'null' , 'null' ,15, 7]


# # 카카오 2021 인턴쉽 코테 4, 5번
# 

# # 미로 탈출

# In[2]:


n= 3
start = 1
end = 3
roads = [[1,2,2],[3,2,3]]
traps = [2]
result = 5


# In[11]:


import heapq as h
def solution(n, start, end, roads, traps):
    start -=1; end -=1;
    INF = float("inf");
    graph = [[] for _ in range(n)] # 문제의 맵 선언 
    trap_dict = {trap-1:idx for idx, trap in enumerate(traps)}; # 이런식으로 dict로 바로 선언가능하구나
    print(trap_dict)
    nodes = [];
    isVisit = [[False]*n for _ in range(1<<len(traps))]
   
    for road in roads:
        start_i, end_i, cost = road
        graph[start_i-1].append([end_i-1,cost,0])
        graph[end_i-1].append([start_i-1,cost,1])
    print(graph)
    h.heappush(nodes,(0,start,0))
    while nodes:
        print(nodes)
        cur_time, cur_node, state = h.heappop(nodes);
        if cur_node == end : return cur_time;      
        if isVisit[state][cur_node] == True: continue;
        else: isVisit[state][cur_node] = True;
            
        for next_node, next_cost, road_type in graph[cur_node]:
            next_state = state
            cur_isTrap = 1 if cur_node in trap_dict else 0;
            next_isTrap = 1 if next_node in trap_dict else 0;

            if cur_isTrap == 0 and next_isTrap == 0:
                if road_type == 1: continue
            elif (cur_isTrap + next_isTrap) == 1:
                node_i = cur_node if cur_isTrap == 1 else next_node
                isTrapOn = (state & (1<<trap_dict[node_i]))>>trap_dict[node_i]
                if isTrapOn != road_type: continue
            else:
                isTrapOn = (state & (1<<trap_dict[cur_node]))>>trap_dict[cur_node]
                n_isTrapOn = (state & (1<<trap_dict[next_node]))>>trap_dict[next_node]
                if (isTrapOn ^ n_isTrapOn) != road_type: continue
            
            if next_isTrap == 1:
                next_state = state ^ (1<<trap_dict[next_node])

            h.heappush(nodes,(cur_time+next_cost, next_node, next_state))


# In[12]:


solution(n, start, end, roads, traps)


# # 시험장 나누기

# In[13]:


k = 3
num = [12,30,1,8,8,6,20,7,5,10,4,1]
links = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[8,5],[2,10],[3,0],[6,1],[11,-1],[7,4],[-1,-1],[-1,-1]]


# In[14]:


import sys
sys.setrecursionlimit(10**6)

l = [0] * 10005 # 왼쪽 자식 노드 번호
r = [0] * 10005 # 오른쪽 자식 노드 번호
x = [0] * 10005 # 시험장의 응시 인원
p = [-1] * 10005 # 부모 노드 번호
n = 0 # 노드의 수
root = 0 # 루트

cnt = 0 # 그룹의 수

# cur : 현재 보는 노드 번호, lim : 그룹의 최대 인원 수
def dfs(cur, lim):
    global cnt
    lv = 0
    if l[cur] != -1: lv = dfs(l[cur], lim)
    rv = 0 # 오른쪽 자식 트리에서 넘어오는 인원 수
    if r[cur] != -1: rv = dfs(r[cur], lim)
    # 1. 왼쪽 자식 트리와 오른쪽 자식 트리에서 넘어오는 인원을 모두 합해도 lim 이하일 경우
    if x[cur] + lv + rv <= lim:
        return x[cur] + lv + rv
    # 2. 왼쪽 자식 트리와 오른쪽 자식 트리에서 넘어오는 인원 중 작은 것을 합해도 lim 이하일 경우
    if x[cur] + min(lv, rv) <= lim:
        cnt += 1 # 둘 중 큰 인원은 그룹을 지어버림
        return x[cur] + min(lv, rv)
    
    # 3. 1, 2 둘 다 아닐 경우
    cnt += 2 # 왼쪽 자식 트리와 오른쪽 자식 트리 각각을 따로 그룹을 만듬
    return x[cur]

def solve(lim):
    global cnt
    cnt = 0
    dfs(root, lim)
    cnt += 1 # 맨 마지막으로 남은 인원을 그룹을 지어야 함
    return cnt

def solution(k, num, links):
    global root
    n = len(num)
    for i in range(n):
        l[i], r[i] = links[i]
        x[i] = num[i]
        if l[i] != -1: p[l[i]] = i
        if r[i] != -1: p[r[i]] = i
    
    for i in range(n):
        if p[i] == -1:
            root = i
            break
    st = max(x)
    en = 10 ** 8
    while st < en:
        mid = (st+en) // 2
        if solve(mid) <= k:
            en = mid
        else: st = mid+1    
    return st


# In[15]:


solution(k, num, links)


# In[ ]:




