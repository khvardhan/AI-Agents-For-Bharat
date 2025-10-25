#!/usr/bin/env python3
"""
Simple .env file validator
Checks if your .env file is properly formatted
"""

from pathlib import Path

print("Checking .env file...")
print()

env_file = Path(".env")

if not env_file.exists():
    print("❌ ERROR: .env file not found!")
    print(f"   Expected location: {env_file.absolute()}")
    print()
    print("Create a .env file with:")
    print("HF_TOKEN=hf_your_token_here")
    exit(1)

print(f"✓ Found .env file at: {env_file.absolute()}")
print(f"  File size: {env_file.stat().st_size} bytes")
print()

# Read and validate
with open(env_file, 'r') as f:
    lines = f.readlines()

print("Analyzing contents...")
print()

found_token = False
issues = []

for i, line in enumerate(lines, 1):
    stripped = line.strip()
    
    # Skip empty lines and comments
    if not stripped or stripped.startswith('#'):
        continue
    
    # Check for token definition
    if 'HF_TOKEN' in line or 'HUGGING_FACE' in line:
        print(f"Line {i}: {stripped[:50]}...")
        
        # Check format
        if '=' not in line:
            issues.append(f"Line {i}: Missing '=' sign")
        elif ' = ' in line:
            issues.append(f"Line {i}: Remove spaces around '=' (should be KEY=value)")
        else:
            parts = line.split('=', 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            
            if not value:
                issues.append(f"Line {i}: Token value is empty!")
            elif not value.startswith('hf_'):
                issues.append(f"Line {i}: Token should start with 'hf_'")
            elif len(value) < 20:
                issues.append(f"Line {i}: Token seems too short (should be ~40 chars)")
            else:
                found_token = True
                print(f"  ✓ Key: {key}")
                print(f"  ✓ Value: {value[:10]}... ({len(value)} chars)")

print()
print("=" * 60)

if issues:
    print("⚠️  ISSUES FOUND:")
    print()
    for issue in issues:
        print(f"  • {issue}")
    print()
    print("Common fixes:")
    print("  • Remove spaces: 'HF_TOKEN=value' not 'HF_TOKEN = value'")
    print("  • No quotes needed: 'HF_TOKEN=hf_xxx' not 'HF_TOKEN=\"hf_xxx\"'")
    print("  • Check token from: https://huggingface.co/settings/tokens")
elif found_token:
    print("✓ .env file looks good!")
    print()
    print("Next steps:")
    print("  1. Install: pip install python-dotenv")
    print("  2. Run: python check_token.py")
    print("  3. Then run your download script")
else:
    print("❌ No HF_TOKEN found in .env file")
    print()
    print("Add this line to your .env file:")
    print("HF_TOKEN=hf_your_token_here")

print("=" * 60)