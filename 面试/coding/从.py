from heapq import heappop, heappush
from typing import List


class Solution:
    def leftmostBuildingQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n=len(nums)
        ans=[-1]*len(queries)
        for idx in range(len(queries)):
            i,j=queries[idx]
            if i>j:
                queries[idx][0],queries[idx][1]=queries[idx][1],queries[idx][0]
                i,j=j,i

            if nums[i]<nums[j] or i==j:
                ans[idx]=j
        h=[]
        queries=[[i,j,idx] for idx,(i,j) in enumerate(queries)]
        queries.sort(key=lambda x:x[1])
        idx=0
        for i in range(n):
            while idx<len(queries) and queries[idx][1]<=i:
                l, r, index = queries[idx]
                if ans[index]==-1:
                    l,r,index=queries[idx]
                    heappush(h,(nums[l],r,index))
                idx+=1
            while h and nums[i]>h[0][0]:
                _,_,pos=heappop(h)
                ans[pos]=i
        return ans
Solution().leftmostBuildingQueries([6,4,8,5,2,7],queries =
[[0,1],[0,3],[2,4],[3,4],[2,2]])