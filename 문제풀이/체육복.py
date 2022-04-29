def solution(n, lost, reserve):
    lost = sorted(lost)
    reserve = sorted(reserve)
    answer = n - len(lost)
    duplicate = []
    for l in lost:
        if l in reserve:
            answer += 1
            loc = reserve.index(l)
            reserve.pop(loc)
            duplicate.append(l)
    for dup in duplicate:
        loc = lost.index(dup)
        lost.pop(loc)
    for l in lost:
        if l - 1 in reserve:
            answer += 1
            loc = reserve.index(l - 1)
            reserve.pop(loc)
            pass
        elif l + 1 in reserve:
            answer += 1
            loc = reserve.index(l + 1)
            reserve.pop(loc)

    return answer


solution(n, lost, reserve)