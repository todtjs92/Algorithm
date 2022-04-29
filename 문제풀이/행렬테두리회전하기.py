def solution(rows, columns, queries):
    matrix = [[y+ columns*x for y in range(1,columns+1)] for x in range(rows)]
    answer= []
    for que in range(len(queries)):
      temp=[]
      for a in range(queries[que][1]-1,queries[que][3]-1):

        temp.append(matrix[queries[que][0]-1][a])

      for b in range(queries[que][0]-1,queries[que][2]-1):

        temp.append(matrix[b][queries[que][3]-1])

      for c in range(queries[que][3]-1,queries[que][1]-1,-1):
        temp.append(matrix[queries[que][2]-1][c])


      for d in range(queries[que][2]-1,queries[que][0]-1,-1):
        temp.append(matrix[d][queries[que][1]-1])

      answer.append(min(temp))
      last = temp.pop()
      temp.insert(0,last)

      count = 0  
      for a in range(queries[que][1]-1,queries[que][3]-1):
        matrix[queries[que][0]-1][a] = temp[count]
        count+=1
      for b in range(queries[que][0]-1,queries[que][2]-1):
        matrix[b][queries[que][3]-1] = temp[count]
        count+=1
      for c in range(queries[que][3]-1,queries[que][1]-1,-1):
        matrix[queries[que][2]-1][c] = temp[count]
        count+=1
      for d in range(queries[que][2]-1,queries[que][0]-1,-1):
        matrix[d][queries[que][1]-1] = temp[count]
        count +=1
    
    return answer
