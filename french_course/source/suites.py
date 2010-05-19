def fib(n):
    "return nth term of Fibonacci sequence"
    a, b = 0, 1
    i = 0
    while i<n:
        a, b = b, a+b
        i += 1 
    return b

def linear_recurrence(n, (a,b)=(2,0), (u0, u1)=(1,1)):
    """return nth term of the sequence defined by the
    linear recurrence
        u(n+2) = a*u(n+1) + b*u(n)"""
    i = 0
    u, v = u0, u1
    while i<n:
        w = a*v + b*u
        u, v = v, w
        i +=1
    return w
        
        
