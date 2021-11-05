from collections import Counter
def minion_game(s):
    Kevin = 0
    Stuart = 0

    for idx, c in enumerate(s):
        if c in "AEIOU":
            Kevin += len(s) - idx
        else:
            Stuart += len(s) - idx

    if Kevin > Stuart:
        print('Kevin', Kevin)
    elif Kevin < Stuart:
        print('Stuart', Stuart)
    else :
        print('Draw')
if __name__ == '__main__':
    s = input()
    minion_game(s)
