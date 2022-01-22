from collections import deque

dp = [[int(1e9) for _ in range(n)] for _ in range(n)]

board = [[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,0,0],[0,0,0,1,0,0,0,1],[0,0,1,0,0,0,1,0],[0,1,0,0,0,1,0,0],[1,0,0,0,0,0,0,0]]

n= len(board)

answer = int(1e9)

dp = [[int(1e9) for _ in range(n)] for _ in range(n)]

directions = [(-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)]  # ( x 좌표 이동 , y좌표 이동 , 디렉션 커브했을때 - - 이면 100원 ,  - | 이면 500원 이 용도임 이걸 찾아 낼 수 있게 direction 심는 용도임)

q = deque([(0, 0, 0, -1)])

while q:
  i, j, cost, dir_idx = q.popleft() # 이사람 한쪽에다 몰아서 넣지 왜 이렇게 넣었지
  if (i, j) == (n - 1, n - 1) and answer > cost:
    answer = cost
  for direction in directions:            # 각 모든 방향 (4 방향) 에 대하여 탐색 
     next_i = i + direction[0]
     next_j = j + direction[1]
     add_cost = 1 if dir_idx == direction[2] or dir_idx == -1 else 6 # 같은 방향이면  cost에 1 추가, 방향 비교해서 없을 경우 유턴이라서 6 추가 
     if not (0 <= next_i < n and 0 <= next_j < n) or board[next_i][next_j] == 1:     # 다음위치가 맵 밖에 나가거나 , 벽에 위치할경우
       continue
     if dp[next_i][next_j] < cost + add_cost - 4:        # 코스트 적으로 손해이면 그쪽으로 탐색하지 않고 continue
       continue  
  dp[next_i][next_j] = cost + add_cost    # 조건들 다 통과하면 기존 코스트 + 추가 코스트를 보드안에 맵핑
  q.append((next_i, next_j, cost + add_cost, direction[2])) # 가야할 곳   에 계속 넣기