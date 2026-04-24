import requests, sys, time
sys.stdout.reconfigure(encoding='utf-8')
job_id = '2bdf6c77-fe00-41ef-848b-ab5816999df1'

for i in range(30):
    time.sleep(30)
    try:
        r = requests.get(f'http://127.0.0.1:5000/trang_thai/{job_id}')
        d = r.json()
        status = d.get('trang_thai')
        print(f"[{(i+1)*30}s] Status: {status}")
        if status in ('hoan_thanh', 'that_bai'):
            print('Error:', d.get('loi', 'none'))
            break
    except Exception as e:
        print(f"[{(i+1)*30}s] Request failed: {e}")

print("=== POLL DONE ===")
