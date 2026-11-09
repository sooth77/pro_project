import subprocess, time, re, os

# Streamlit 실행 (터널 주소로 들어오는 접속을 막지 않도록 CORS·XSRF 검사를 끔, 로그는 파일로 남겨 문제가 생기면 확인)
st_proc = subprocess.Popen(
    ["streamlit", "run", "frontend/app_chatbot.py", "--server.port", "8501", "--server.headless", "true",
     "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"],
    stdout=open("streamlit.log", "w"), stderr=subprocess.STDOUT)
time.sleep(5)
print("Streamlit 응답:", subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "localhost:8501"],
                                        capture_output=True, text=True).stdout)   # 200이면 정상

# cloudflared 설치(없을 때만) 후 8501번만 공개 (가입 불필요한 임시 주소)
if subprocess.run(["which", "cloudflared"], capture_output=True).returncode != 0:
    subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
                   " && dpkg -i cloudflared-linux-amd64.deb > /dev/null", shell=True)
tunnel = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://localhost:8501"],
                          stdout=open("tunnel.log", "w"), stderr=subprocess.STDOUT)
time.sleep(10)
print("공개 주소:", re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", open("tunnel.log").read()))
