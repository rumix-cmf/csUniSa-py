def print_result(test_name, status, error=None):
    """
    Print formatted test result with ANSI colours and status.

    Parameters
    ----------
    test_name : str
        Label for the test.
    status : str
        One of: "PASS", "WARN", "FAIL".
    error : float
        Max absolute error.
    """
    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"

    if status == "PASS":
        colour = GREEN
        symbol = "✅"
    elif status == "WARN":
        colour = YELLOW
        symbol = "⚠️ "
    else:
        colour = RED
        symbol = "❌"

    print(f"{test_name:<40} {colour}{symbol} {status}{RESET}")
    if error is not None:
        print(f"{colour}Max error: {error:.2e}{RESET}")