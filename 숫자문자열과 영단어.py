# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GR9QIxQqDb544XoI15pp4KdnfFDxFWEi
"""

def solution(s):
    numbers = ['zero' ,'one','two' ,'three' ,'four' ,'five' ,'six' , 'seven', 'eight' ,'nine']
    answer = 0
    previous = s
    post = 'random'
    while previous != post:
      post = previous
      for num, alpha in enumerate(numbers):
        previous = previous.replace(alpha,str(num))
    answer = int(previous)
    return answer