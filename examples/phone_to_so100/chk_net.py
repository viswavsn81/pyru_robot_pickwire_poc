import hebi
import time

print("Scanning for HEBI Mobile I/O devices (5 seconds)...")
lookup = hebi.Lookup()

# Wait for discovery
time.sleep(5)

count = 0
print("\n--- RESULTS ---")

# Iterate through whatever we found
for entry in lookup.entrylist:
    count += 1
    print(f"✅ FOUND DEVICE #{count}")
    print(f"   Name:   '{entry.name}'")
    print(f"   Family: '{entry.family}'")
    print("   -----------------------")

if count == 0:
    print("❌ NO devices found.")
    print("   Possibilities:")
    print("   1. App is not open/active on screen.")
    print("   2. Firewall/Router is blocking broadcast packets.")
    print("   3. Try the 'Personal Hotspot' trick.")
