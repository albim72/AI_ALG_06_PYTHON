import inspect

def compute(x:int, y:int=10)->int:
    return x + y

print(compute(1, 2))
print(inspect.signature(compute))
print(inspect.getdoc(compute))
print(inspect.getsource(compute))
print(inspect.isfunction(compute))
