# -*- coding: utf-8 -*-
"""표편집.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16fEDr1ORiWVMnYO6EYk30-mZJPq_V45u
"""

n=  8
K = 2

cmd = ["D 2","C","U 3","C","D 4","C","U 2","Z","Z"]

# 처음시도 
# 이렇게 할게아니라 [0,1,0,0,0,1] 식으로 표시해주고 삭제한 원소들을 스택에 넣어줬어야는데

start_list = list(range(n))
now_list = list(range(n))
now_name_list = list(range(n))
now_counter = K

import copy

for i in cmd:
  if len(i)>1:
    command , number = i.split(' ')
    if command == 'U':
      previous_counter = now_counter
      now_counter = now_counter - number

    else:
      previous_counter = now_counter
      now_counter = now_counter + number
  else:
    if i == 'C':
      previous_list = now_list
      removed_item = now_list.pop(now_counter)
      removed_name = now_name_list.pop(now_counter)
    else:
      now_list = now_list.insert(now_counter , )
      if

import collections

n=  8
k = 2

cmd = ["D 2","C","U 3","C","D 4","C","U 2","Z","Z"]

answer = ["O" for i in range(n)]
linked_list = collections.defaultdict(list)

for i in range(1, n + 1):
  linked_list[i].append(i - 1)
  linked_list[i].append(i + 1)

# 왼쪽 , 오른쪽 연결시켜놈
linked_list





cmd = ["D 2","C","U 3","C","D 4","C","U 2","Z","Z"]

# linked_list 방식

from collections import defaultdict


def solution(n, k, cmd):
    answer = ["O" for i in range(n)]
    linked_list = defaultdict(list)

    for i in range(1, n + 1):
        linked_list[i].append(i - 1)
        linked_list[i].append(i + 1)

    stack = []
    k += 1
    print(linked_list)
    for instruction in cmd:
        if instruction[0] == "D":                 # 링크드리스트를 하나 하나 움직이는 방식 , 중간에 고리가 빠져있기 떄문에 어쩔수 없는거 같긴한데 이래도 테스트를 통과하구나 
            for _ in range(int(instruction[2:])):
                k = linked_list[k][1]
                print(k , 'Down')
        elif instruction[0] == "U":
            for _ in range(int(instruction[2:])):
                k = linked_list[k][0]
                
        elif instruction[0] == "C":             # prev 와 next를 동시에 저장 
            prev, next = linked_list[k]         # stack에다 리스트 통쨰로 넣어줌 . 
            stack.append([prev, next, k])
            answer[k - 1] = "X"

            if next == n + 1:
                k = linked_list[k][0]
            else:
                k = linked_list[k][1]

            if prev == 0:
                linked_list[next][0] = prev
            elif next == n + 1:
                linked_list[prev][1] = next
            else:
                linked_list[prev][1] = next
                linked_list[next][0] = prev

        elif instruction[0] == "Z":
            prev, next, now = stack.pop()     # 요렇게 팝하기만하면됨 .
            answer[now - 1] = "O"

            if prev == 0:
                linked_list[next][0] = now
            elif next == n + 1:
                linked_list[prev][1] = now
            else:
                linked_list[prev][1] = now
                linked_list[next][0] = now

    return "".join(answer)

solution(n, k, cmd)

cmd = ["D 2","C","U 3","C","D 4","C","U 2","Z","Z"]

# heapq 구현
# 포인터가 항상 리스트에 끝에있어서 넣는 속도가 빠름

import heapq

left, right ,delete =[] , [], []

for i in range(n):
  heapq.heappush(right,i)
  print(right)

for i in range(k):
  heapq.heappush(left , -heapq.heappop(right))
  print(left, i)

right

left

