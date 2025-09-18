# compare_large_ids.py
import os

def load_ids(path):
    """Load IDs from a text file into a set, stripping whitespace."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return {line.strip() for line in f if line.strip()}

file1 = r"C:\Users\Ken\Documents\GitHub\Echo-Me\all_words_other.txt"
file2 = r"C:\Users\Ken\Documents\GitHub\Echo-Me\all_words_new.txt"

ids1 = load_ids(file1)
ids2 = load_ids(file2)

only_in_file1 = ids1 - ids2
only_in_file2 = ids2 - ids1
in_both = ids1 & ids2

# Write full report without limiting number of IDs
with open("differences_report.txt", "w", encoding="utf-8") as f:
    f.write("=== Only in file1 ===\n")
    f.write("\n".join(sorted(only_in_file1)) + "\n\n")

    f.write("=== Only in file2 ===\n")
    f.write("\n".join(sorted(only_in_file2)) + "\n\n")

    f.write("=== In both files ===\n")
    f.write("\n".join(sorted(in_both)) + "\n")

print("Done ✅ Differences written to differences_report.txt")
print(f"File1 unique IDs: {len(ids1)}")
print(f"File2 unique IDs: {len(ids2)}")
print(f"Only in file1: {len(only_in_file1)}")
print(f"Only in file2: {len(only_in_file2)}")
print(f"In both: {len(in_both)}")
