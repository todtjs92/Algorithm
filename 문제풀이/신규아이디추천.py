import re 
def solution(new_id): 
    #1
    new_id=new_id.lower() 
    #2
    lvl2 = re.compile('[0-9a-z_.\-]+') 
    new_id = lvl2.findall(new_id) 
    new_id = ''.join(new_id) 
    #3
    while '..' in new_id: 
        new_id = new_id.replace('..','.') 
    #4
    new_id = new_id.strip('.') 
    
    #5
    if new_id =='':new_id+='a' 
    #6
    if len(new_id)>=16: 
        new_id = new_id[:15] 
        new_id = new_id.rstrip('.')
    
    #7
    if len(new_id)<=2: 
        idSize = len(new_id) 
        addchar = new_id[idSize-1:] 
        while len(new_id)<3:
            new_id+=addchar 
    answer = new_id 
    return answer

