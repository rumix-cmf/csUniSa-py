import unit_tests.problems as ivps

from csunisa.lmm import LinearMultistepMethod
from csunisa.lmm_registry import get_method, list_methods
from unit_tests.tester import test_lmm


def unit_test_lmms(h=0.01):
    methods = list_methods()

    for method in methods:
        data = get_method(method)
        lmm = LinearMultistepMethod(data["alpha"], data["beta"], method)
        print(f"🧪 Running: {lmm.name}")
        print("-" * 50)
        for pb_function in ivps.lmm_problems.values():
            pb = pb_function()
            test_lmm(lmm, pb, h)
        print("\n")


def main():
    unit_test_lmms()


if __name__ == "__main__":
    main()
