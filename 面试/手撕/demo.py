'''
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。
[-1,0,1,2,-1,-4]
'''

def main(nums):
    nums.sort()
    n=len(nums)
    ans=[]
    for i in range(n-2):
        l,r=i+1,n-1
        target=-nums[i]
        if i and nums[i]==nums[i-1]:
            continue
        while l<r:
            tmp=nums[l]+nums[r]
            if tmp==target:
                ans.append([nums[i],nums[l],nums[r]])
                l+=1
                r-=1
            elif tmp>target:
                r-=1
            else:
                l+=1
    print(ans)

nums=[-1,0,1,2,-1,-4]
main(nums)