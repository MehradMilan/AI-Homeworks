# import numpy as np
import copy
import time

class CSP:
    def __init__(self, number_of_marks):
        self.number_of_marks = number_of_marks
        self.current_length = int(number_of_marks * (number_of_marks - 1) // 2)
        self.variables = [] * number_of_marks
        self.differences = []
        for _ in range(number_of_marks):
            self.differences.append([])
        
    def assign_value(self, i, v, differences):
        self.variables.append(v)
        for j in range(i):
            differences[j].append(abs(v - self.variables[j]))        

    def check_constraints(self, i, v, differences):
        for j in range(i):
            diff = abs(self.variables[j] - v)
            for k in range(i):
                if(diff in differences[k]):
                    return False
        return True
        

    def backtrack(self, i):
        if i == self.number_of_marks:
            return True
        for d in self.get_domain_range(i):
            if self.check_constraints(i, d, self.differences):
                self.assign_value(i, d, self.differences)
                if(self.backtrack(i+1)):
                    return True
                else:
                    self.variables.pop()
                    self.update_differences(i)                    
        return False

    def update_differences(self, i):
        for j in range(i):
            self.differences[j].pop()

    def get_domain_range(self, i):
        if i == 0:
            return range(0, self.current_length + 1)
        else:
            return range((self.variables[i-1] + 1), (self.current_length + 1)) 

    def find_minimum_length(self) -> int:
        while True:
            if(self.backtrack(0)):
                return self.current_length
            self.current_length += 1

    def get_variables(self) -> list:
        return self.variables