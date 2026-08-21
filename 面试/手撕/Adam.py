import numpy as np

# x = np.array([(1, 1), (1, 2), (2, 2), (3, 1), (1, 3), (2, 4), (2, 3), (3, 3)])  # (x1, x2) (8,2)  np.random.randn(m, n)
# x1=x[:,0]
# x2=x[:,1]
# y=3*x1+2*x2+np.random.randn(len(x))
#
# b1=0.9
# b2=0.999
# lr=0.05
# e=1e-8
#
# theta=np.zeros(x.shape[1])
# m=len(y)
#
# m_t=np.zeros_like(theta)
# v_t=np.zeros_like(theta)
#
# def adam():
#     global theta,m_t,v_t
#     m=x.shape[0]
#     for t in range(1,101):
#         pred=x@theta
#         grad=(1/m)*x.T@(pred-y)
#
#         m_t=b1*m_t+(1-b1)*grad
#         v_t=b2*v_t+(1-b2)*grad**2
#
#         m_t_hat=m_t/(1-b1**t)
#         v_t_hat=v_t/(1-b2**t)
#
#         theta-=lr/(np.sqrt(v_t_hat)+e)*m_t_hat
#         if t % 50 == 0:
#             loss = np.mean((pred - y)**2)/2
#             print(f"iter {t}, loss {loss:.4f}, theta {theta}")
#
# adam()
eps=1e-8
x=np.random.randn(9,2)
x1,x2=x[:,0],x[:,1]
y=x1*3+x2*4+np.random.randn(x.shape[0])

b1,b2=0.9,0.99
lr=0.01

theta=np.zeros(x.shape[-1])
m=np.zeros_like(theta)
v=np.zeros_like(theta)

for t in range(1,1001):
    pred=x@theta
    grad=1/len(y)*x.T@(pred-y)
    m=b1*m+(1-b1)*grad
    v=b2*v+(1-b2)*grad**2

    m_t=m/(1-b1**t)
    v_t=v/(1-b2**t)

    theta-=lr/(np.sqrt(v_t)+eps)*m_t

    if t % 50 == 0:
        loss = np.mean((pred - y)**2)/2
        print(f"iter {t}, loss {loss:.4f}, theta {theta}")
