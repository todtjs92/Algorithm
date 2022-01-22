import collections
color_dict = collections.defaultdict(int) {1: 0}

def bfs(i,j):
  
  if i < 0 or i>= len(picture)  or j < 0 or j >= len(picture[0]) or picture[i][j]!=color:
    return
  color_dict[color]+=1
  print(i,j)
  picture[i][j]=0
  bfs(i-1,j)
  bfs(i+1,j)
  bfs(i,j+1)
  bfs(i,j-1)


for i in range(len(picture)):
  for j in range(len(picture[0])):
    if picture[i][j] != 0:
      color = picture[i][j]
      if final_dict[str(color)] != 0:
        final_dict[str(color)+str(color)]
        previous = color_dict[color]
		{1:5 ,11:}
      bfs(i,j)
      if final_dict[str(color)] != 0:
        final_dict[str(color)+str(color)]  = color_dict[color] - previous
      else:
        final_dict[str(color)] = color_dict[color]
print(len(final_dict))
print(collections.Counter(final_dict).most_common(1)[0][1])

