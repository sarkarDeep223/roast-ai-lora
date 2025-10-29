
class ListNode:
    def __init__(self,val=0,next=None):
        self.val = val
        self.next = next;



class Solution :
    def sumOfTwo(self,nums,terget):
        num_set = {};
        for i,num in  enumerate(nums):
            terget_num = terget - num;
            if terget_num in num_set:
                return [i,num_set[terget_num]]
            num_set[num] = i;
        return [];


    def isPolindrome(self,x):
        if x < 0:
            return False;

        original = x;
        reversed_num = 0;

        while x != 0:
            digit = x % 10;
            reversed_num = reversed_num * 10 + digit;
            x = x // 10;
        if original == reversed_num:
            return True;
        else:
            return False;

    def tomanToInt(self,s):
        roman_numerals = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        };

        total = 0;

        for i in range(len(s)):
            if i + 1 < len(s) and roman_numerals[s[i]] < roman_numerals[s[i+1]]:
                total -= roman_numerals[s[i]];
            else:
                total += roman_numerals[s[i]];
        return total;





    def longestCommonPrefix(self, strs):
        if not strs:
            return False;

        prefix  = strs[0];


        for s in strs[1:]:
            while not s.startsWith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return "";
        return prefix;


    def isValidParentheses(self,s):

        stack = [];
        mapping  = {")":"(", "}":"{", "]":"["};

        for char in s:
            if char in mapping:
                if stack and stack[-1] == mapping[char]:
                    stack.pop();
                else:
                    return False;
            else:
                stack.append(char);
        return not stack;



    def mergeTwoLists(self,l1,l2):
        dummy = ListNode();
        current = dummy;
        while l1 and l2:
            if l1.val < l2.val:
                current.next = l1;
                l1 = l1.next;
            else:
                current.next = l2;
                l2 = l2.next;
            current = current.next;
        current.next = l1 if l1 else l2;
        return dummy.next;



s = Solution()
print(s.sumOfTwo([2,7,11,12,15,25,55,62], 40))