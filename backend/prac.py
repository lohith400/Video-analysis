# def is_palindrome(s: str) -> bool:
#     # Write your logic here
#     #s = str
#     # clean = s.replace(" ", "").replace(",", "")
#     # rev = clean[::-1]
#     # print(clean,rev)
#     clean = ""
#     for char in s:
#         if (char.isalnum()):
#             clean += char.lower()

#     if (clean == clean[::-1]):
#         return True
#     else:
#         return False


# # Optional: Add test cases to verify
# print(is_palindrome("A man, a plan, a canal: Panama"))  # Expected: True
# print(is_palindrome("race a car"))                      # Expected: False



# def two_sum(nums: list[int], target: int) -> list[int]:
#     # Write your logic here
#     nums = sorted(nums)
#     left = 0
#     right = len(nums) -1
#     while left < right:
#         if nums[left] + nums[right] == target:
#             return left,right
#         elif nums[left] + nums[right] > target:
#             right-=1
#         elif nums[left] + nums[right] < target:
#             left+=1



# # Test case
# print(two_sum([2, 7, 11, 15], 9))  # Expected output: [0, 1] (or [1, 0])
# print(two_sum([3, 2, 4], 6))       # Expected output: [1, 2]

# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

# def solution(N):
#     # Implement your solution here

#     str_N = bin(N)
#     print(str_N)
#     first = len(str_N)
#     ans = 0
#     for i in range(2,len(str_N)):
#         if str_N[i] == '1':
#             if first < i:
#                 new = i-first-1
#                 if ans <= new:
#                     ans = new
#                     # first = i
#                 # elif new == 0:
#                 first = i
#             else:
#                 first = i
        


#     return ans

# print(solution(5664))

# print(1%5)

# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

# def solution(A, K):
#     # Implement your solution here
#     ans_list = A.copy()
#     for i in range(K):
#         if not A:
#             return A
#         else:
#             A = ans_list.copy()
#             for j in range(0,len(A)):
#                 if j == len(A)-1:
#                     ans_list[0] = A[-1]
#                 else:
#                     ans_list[j+1] = A[j] 

#     return ans_list
        

# print(solution([3, 8, 9, 7, 6],3))  


def solution(A):
    # Implement your solution here
    set_A = dict()
    for i in set(A):
        set_A[i] = A.count(i)

    return min(set_A, key=set_A.get)

print(solution([9,3,9,3,9,7,9]))

    
        
