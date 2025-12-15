import os
import re

def get_p_value(filename):
    match = re.search(r'_p(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return -1

def generate_typst():
    files = [f for f in os.listdir('.') if f.endswith('.png')]
    
    box_files = []
    gauss_files = []

    for f in files:
        p = get_p_value(f)
        if p == -1: continue
        
        if 'box' in f:
            box_files.append((p, f))
        elif 'gaussian' in f:
            gauss_files.append((p, f))
            
    # Sort by p-value
    box_files.sort(key=lambda x: x[0])
    gauss_files.sort(key=lambda x: x[0])
    
    lines = []
    lines.append('#set page(')
    lines.append('  fill: black,')
    lines.append('  margin: 0.5cm,')
    lines.append('  height: auto,')
    lines.append(')')
    lines.append('#set text(fill: white, font: "New Computer Modern", size: 12pt)')
    lines.append('')
    
    # Box Blur
    lines.append('#grid(')
    lines.append('  columns: (1fr, 1fr, 1fr, 1fr),')
    lines.append('  gutter: 5pt,')
    lines.append('  align: center,')
    
    for p, filename in box_files:
        lines.append(f'  stack(dir: ttb, spacing: 5pt, image("{filename}", width: 100%), text([p = {p}])),')
        
    lines.append(')')
    lines.append('')
    lines.append('#pagebreak()')
    lines.append('')
    
    # Gaussian Blur
    lines.append('#grid(')
    lines.append('  columns: (1fr, 1fr, 1fr, 1fr),')
    lines.append('  gutter: 5pt,')
    lines.append('  align: center,')
    
    for p, filename in gauss_files:
        lines.append(f'  stack(dir: ttb, spacing: 5pt, image("{filename}", width: 100%), text([p = {p}])),')
        
    lines.append(')')

    typst_content = '\n'.join(lines)

    with open('visualization.typ', 'w') as f:
        f.write(typst_content)
        
    print("Generated visualization.typ")

if __name__ == "__main__":
    generate_typst()