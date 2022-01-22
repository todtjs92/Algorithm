# cmd = ["D 2","C","U 3","C","D 4","C","U 2","Z","Z"]
# n=  8
# K = 2

from collections import defaultdict

def solution(n,k,cmd):
    answer = ["0" for i in range(n)]
    linked_list = defaultdict(list) 
    
    for i in range(1, n+1):            # 이전 , 이후 원소 값 저장해놈 , append통해 자동으로 생성가능
        linked_list[i].append(i-1)
        linked_list[i].append(i+1)

    stack=[]     # stack에는 지운것들 들어갈 예정
    k+=1

    for instruction in cmd:
        if instruction in cmd:
            if instruction[0] == "D":
                for _ in range(int(instruction[2:])):  # linked_list로 연결 됬기때문에 for문 돌면서 이동 
                    k = linked_list[k][1]
            elif instruction[0] =="U":
                for _ in range(int(instruction[2:])):
                    k = linked_list[k][0] 
                   
            elif instruction[0] == "C":
                prev, next = linked_list[k]
                stack.append([prev,next,k])
                answer[k-1] = "X"

                if  next == n+1:
                    k = linked_list[k][0]
                else:
                    k = linked_list[k][1]

                if prev == 0:
                    linked_list[next][0] = prev
                elif next == n+1:
                    linked_list[prev][1] = next
                    linked_list[next][0] = prev
        elif instruction[0]=="z":
            prev,next,now = stack.pop()
            answer[now -1]= "o" 
            if prev ==0:
                linked_list[next][0] = now
            elif next== n+1:
                linked_list[prev][1] = now
            else:
                linked_list[prev][1] = now
                linked_list[next][0] = now                
            
    return "".join(answer)	    
	

