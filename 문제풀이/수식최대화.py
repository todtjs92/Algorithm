from collections import deque
def bfs(p):
  start = []

  for i in range(5):
    for j in range(5):
      if p[i][j] =='P':
        start.append([i,j]) # P인지점을 start라는 리스트에 다 넣엊두고 

  for s in start: # 모든 start에 대해서 맵을 계속 다시 만드는 방법으로 제작 .
    queue = deque([s])
    visited = [[0]*5 for i in range(5)]
    distance = [[0]*5 for i in range(5)] # distance 라고 따로 거리 체크하는 맵을 한개 더만듬 
    visited[s[0]][s[1]] = 1  # 방문시 visited 라는 맵에 1로 표시 

    while queue:
      y, x =  queue.popleft()

      dx = [-1, 1, 0, 0 ]
      dy = [0,0,-1,1]

      for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]

        if 0 <= nx <5 and 0 <= ny < 5 and visited[ny][nx] ==0:
          if p[ny][nx] == 'O':
            queue.append([ny,nx])
            visited[ny][nx] = 1
            distance [ny][nx] = distance[y][x] + 1

          if p[ny][nx] == 'P' and distance[y][x] <=1:
            return 0
  return 1
def solution(places):
  answer = []

  for i in places:
    answer.append(bfs(i))
  return answer
