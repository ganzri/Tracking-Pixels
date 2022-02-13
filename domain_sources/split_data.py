"""
Script can be used to split a Tranco List into subsets, to then run the presence crawl only on a subset. 
"""

set_a = set()
set_b = set()
set_c = set()

# Load Tranco dataset to split
count = 0

with open("./Tranco_Lists/new_domains.txt", 'r') as fd:
    for line in fd:
        if count%3 == 0:
            set_a.add(line)
        elif count%3 == 1:
            set_b.add(line)
        else:
            set_c.add(line)
        count += 1

print(f"Num total: {count}")
print(f"set a: {len(set_a)}")
print(f"set b: {len(set_b)}")
print(f"set c: {len(set_c)}")

print(len(set_b - set_a))
print(len(set_c-set_b))
print(len(set_a-set_c))


with open("./Tranco_Lists/set_a_europe.txt", 'w') as fd:
    for n in set_a:
        fd.write(n)

with open("./Tranco_Lists/set_b_europe.txt", 'w') as fd:
    for n in set_b:
        fd.write(n)

with open("./Tranco_Lists/set_c_europe.txt", 'w') as fd:
    for n in set_c:
        fd.write(n)

