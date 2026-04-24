import sys
try:
    import bs4
    with open("check_bs4.txt", "w") as f:
        f.write(f"SUCCESS: {bs4.__version__}")
except Exception as e:
    with open("check_bs4.txt", "w") as f:
        f.write(f"FAILED: {str(e)}")
