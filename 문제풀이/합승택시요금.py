
import collections
import heapq

n= 6
s= 4
a = 6
b = 2

fares = [[4,1,10],[3,5,24],[5,6,2],[3,1,41],[5,1,24],[4,6,50],[2,4,66],[2,3,22],[1,6,25]]

graph = collections.defaultdict(list)

for u,v,w in fares:
  graph[u].append((v,w))

graph

n= 6
src= 4
dst = 6
b = 2



"""# 다익스트라"""

import heapq

def solution(n, s, a, b, fares):

    link = [[] for _ in range(n+1)]
    for x, y, z in fares:
        link[x].append((z, y))
        link[y].append((z, x))

    def dijkstra(start):
        dist = [987654321] * (n + 1)
        dist[start] = 0
        heap = []
        heapq.heappush(heap, (0, start))
        print(dist,'dist')
        while heap:
            print(heap)
            value, destination = heapq.heappop(heap)
            if dist[destination] < value:
                continue

            for v, d in link[destination]:
                next_value = value + v
                if dist[d] > next_value:
                    dist[d] = next_value
                    heapq.heappush(heap, (next_value, d))
        return dist

    dp = [[]] + [dijkstra(i) for i in range(1, n+1)]
    # print(dp)
    answer = 987654321
    for i in range(1, n+1):
        answer = min(dp[i][a] + dp[i][b] + dp[i][s], answer)

    return answer

solution(n, s, a, b, fares)



link = [[] for _ in range(n+1)]
for x, y, z in fares:
    link[x].append((z, y))
    link[y].append((z, x))

link #0 은 아무것도 1은 4에 가격은 10 요런식으로 넣음 중복으로



"""# 플로이드 와샬"""

n= 6
s= 4
a = 6
b = 2

fares = [[4,1,10],[3,5,24],[5,6,2],[3,1,41],[5,1,24],[4,6,50],[2,4,66],[2,3,22],[1,6,25]]

INF = int(1e9)
graph = [[INF] * n for _ in range(n)]

graph

import sys
input = sys.stdin.readline
def solution(n, s, a, b, fares):

  # 그래프 세팅 하는 부분
  INF = int(1e9)                                  #무한을 의미하는 값 10억 설정
  graph = [[INF] * n for _ in range(n)]			
  for i in range(n):                              #자기 자신으로 가는 비용 0
    graph[i][i] = 0
  for i in fares:
    graph[i[0] - 1][i[1] - 1] = i[2]            #이동 방향에 따라 비용이 달라지지 않으므로
    graph[i[1] - 1][i[0] - 1] = i[2]			


  # 

  for t in range(n):
    for i in range(n):
      for j in range(i, n): # 0 - 0,1,2,3,4,5 /  1- 1,2,3,4,5,/ 2,3,4,5 / 3,4,5                  
        if i != j:                          #최소 비용 계산
          temp = min(graph[i][j], graph[i][t] + graph[t][j])  # i-j 바로 연결과 어느 곳을 경유 했을 경우를 비교해서 작은 값을 넣는 과정이군.
          graph[i][j] = graph[j][i] = temp


  # S-T , T-A , T-B 로 구분하기
  answer = INF
  for t in range(n):                              #경유점에 따라 최소 합승 비용 탐색
    temp = graph[s - 1][t] + graph[t][b - 1] + graph[t][a - 1]      
    answer = min(answer, temp)
  return answer

solution(n,s,a,b,fares)



for t in range(n):
  for i in range(n):
    for j in range(i,n):
      if i != j:
        temp = min(graph)

