import os

def greet(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    user = os.environ.get("USER", "world")
    print(greet(user))
