import os
import glob

TEST_DIR = os.path.join(os.path.dirname(__file__), 'tests')


def main():
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, 'test_*.py')))
    if not test_files:
        print("No test files found.")
        return

    print("\n🔍 Running all solver tests\n")
    print("=" * 40)

    for test_file in test_files:
        print(f"🧪 Running: {os.path.basename(test_file)}")
        exit_code = os.system(f"python {test_file}")
        print("-" * 40)
        if exit_code != 0:
            print(f"❌ {os.path.basename(test_file)} failed.\n")
        else:
            print(f"✅ {os.path.basename(test_file)} passed.\n")


if __name__ == "__main__":
    main()
