numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
hand = 'right'
left_side = [1,4,7,'*']
right_side = [3,6,9,'#']
mid_side = [2,5,8,0]
phone=[left_side,mid_side, right_side]
answer = ''
left_v , left_h  = 0,3
right_v , right_h  = 2,3
for i in numbers:
  if i in left_side:
    left_v = 0
    left_h = phone[0].index(i)
    answer+= 'L'

  elif i in right_side:
    right_v = 2
    right_h = phone[2].index(i)
    answer+='R'
  else:
    loc =phone[1].index(i)
    if abs(left_v - 1) + abs(left_h - loc) > abs(right_v - 1) + abs(right_h - loc):
      right_v , right_h = 1, loc
      answer+='R'
    elif abs(left_v - 1) + abs(left_h - loc) < abs(right_v - 1) + abs(right_h - loc):
      left_v , left_h = 1, loc
      answer+='L'
    else:
      if hand == 'right':
        right_v , right_h = 1, loc
        answer+='R'
      else:
        left_v , left_h = 1, loc
        answer+= 'L'
  print(left_v, left_h , right_v,right_h           ,'!!!!',i)
print(answer)