## dcnv2

1. 基于dcnnew，删除top_read_books、reapunch特征
2. 删除senet
3. 新用户searchwords为空，likemarks不为空的时候将likemarks填充进searchwords，离线测试能大幅提高推荐准确率
4. 新增week特征