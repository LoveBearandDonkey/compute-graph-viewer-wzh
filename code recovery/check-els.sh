#!/bin/bash
# Verify all els.X references in app.js are registered in the els object
# Run after any modification to app.js

cd "$(dirname "$0")"

python3 -c "
import re, sys

with open('src/app.js') as f:
    js = f.read()

# Find els object: from 'const els = {' to matching '};'
start = js.find('const els = {')
if start < 0:
    print('ERROR: const els = { not found')
    sys.exit(1)

# Find matching }; by counting braces
depth = 0
end = start
for i in range(start, len(js)):
    if js[i] == '{': depth += 1
    elif js[i] == '}': 
        depth -= 1
        if depth == 0:
            end = i
            break

els_block = js[start:end+1]

# Extract registered keys
registered = set(re.findall(r'(\w+):\s*byId\(', els_block))
registered.update(re.findall(r'(\w+):\s*Array\.from\(', els_block))
registered.update(re.findall(r'(\w+):\s*document\.querySelector', els_block))
# Common non-byId keys
registered.update(['executionTabs', 'frameActions', 'tensorCanvas', 'tensorFallback',
    'tensorStage', 'architectureViewportRoot', 'architectureViewport',
    'tensorTabs', 'tensorDataDumpPanel', 'tensorSection'])

# Find els.X usages outside els block
# Exclude common false positives (Array methods, etc.)
false_positives = {'find', 'map', 'push', 'filter', 'forEach', 'length', 'call', 'apply', 'bind'}

issues = []
for m in re.finditer(r'els\.(\w+)', js):
    pos = m.start()
    key = m.group(1)
    if pos > start and pos < end:
        continue  # inside els initialization
    if key in false_positives:
        continue
    if key not in registered:
        line = js[:pos].count('\n') + 1
        issues.append(f'  line {line}: els.{key}')

if issues:
    print('UNREGISTERED els.X references:')
    for i in issues:
        print(i)
    print(f'\n{len(issues)} unregistered reference(s).')
    print('Fix: add \"{key}: byId(\\'{key}\\'),\" to const els = {{ ... }}')
    sys.exit(1)
else:
    print('OK: all els.X references are registered')
    sys.exit(0)
"
