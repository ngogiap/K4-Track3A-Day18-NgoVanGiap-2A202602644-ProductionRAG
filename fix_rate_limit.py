import re

# 1. naive_baseline.py
with open('naive_baseline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add sleep before completions.create
pattern1 = r'(resp = llm_client\.chat\.completions\.create\()'
replacement1 = r'import time; time.sleep(4.5)\n                \1'
content = re.sub(pattern1, replacement1, content)

with open('naive_baseline.py', 'w', encoding='utf-8') as f:
    f.write(content)


# 2. src/pipeline.py
with open('src/pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern2 = r'(resp = client\.chat\.completions\.create\()'
replacement2 = r'import time; time.sleep(4.5)\n            \1'
content = re.sub(pattern2, replacement2, content)

with open('src/pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)


# 3. src/m5_enrichment.py
with open('src/m5_enrichment.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern3 = r'(response = client\.chat\.completions\.create\()'
replacement3 = r'import time; time.sleep(4.5)\n            \1'
content = re.sub(pattern3, replacement3, content)

with open('src/m5_enrichment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched all files for rate limiting!")
