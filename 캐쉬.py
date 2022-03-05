from collections import deque
def solution(cacheSize, cities):
    cities = map(lambda x: x.lower(), cities)
    cache= []
    answer= 0
    if cacheSize ==0:
        answer = 5* len(list(cities))
    else:
        for i in cities:
            if i in cache:
                answer+=1
                inx = cache.index(i)
                item =cache.pop(inx)
                cache.append(item)
            else:
                answer+=5
                if len(cache) != cacheSize:
                    cache.append(i)
                else:
                    cache.pop(0)
                    cache.append(i)
       
    
    return answer
