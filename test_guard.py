from dich_vu.openai_da_buoc import nhom_thuat_ngu_va_tao_dan_y, InsufficientDataError

def test_architect_guard():
    print("Testing Architect Guard with 0 terms...")
    try:
        # Simulation of empty terms result
        nhom_thuat_ngu_va_tao_dan_y("Chủ đề test", [], [], quy_mo="tieu_chuan")
        print("FAILED: Should have raised InsufficientDataError")
    except InsufficientDataError as e:
        print(f"PASSED: Caught error: {e}")
    except Exception as e:
        print(f"FAILED: Caught wrong error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_architect_guard()
