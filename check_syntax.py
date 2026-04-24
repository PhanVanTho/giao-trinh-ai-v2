import ast, os, sys, traceback

def check_syntax():
    with open('syntax_errors.txt', 'w', encoding='utf-8') as out:
        has_error = False
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.py'):
                    path = os.path.join(root, f)
                    with open(path, 'r', encoding='utf-8') as file:
                        source = file.read()
                    try:
                        ast.parse(source, filename=path)
                    except Exception as e:
                        out.write(f"Error in {path}: {e}\n")
                        has_error = True
        
        # Also check for common undefined variables simply by compiling
        # Not a full linter, but finds syntax/indentation errors
        if not has_error:
            out.write("No syntax errors found.\n")

check_syntax()
